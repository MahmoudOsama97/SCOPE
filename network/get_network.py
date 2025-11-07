from network import ResNet
import torch.nn as nn
from utils.lora_util import inject_trainable_moe_kronecker_new
import timm
import torch
from typing import Union, List, Tuple
from torch.autograd import Function

from collections import OrderedDict
import os
class GradientReversalLayerFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features, num_domains):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_domains)
        )
    def forward(self, x):
        return self.network(x)

class Bottleneck(nn.Module):
    def __init__(self, in_features, bottleneck_dim=256):
        super().__init__()
        self.encoder_mu = nn.Linear(in_features, bottleneck_dim)
        self.encoder_logvar = nn.Linear(in_features, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, in_features)
        self.kl_loss = 0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        
        # Calculate KL divergence
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        return self.decoder(z)

class FedWrapper(nn.Module):
    """
    A robust wrapper that uses PyTorch hooks to extract intermediate features,
    guaranteeing that the model's original, high-performance forward pass is
    never altered.
    """
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.private_blocks_count = args.private_blocks_count
        print("\n[DEBUG] Initializing FedWrapper...")
        print(f"[DEBUG] Private blocks count: {self.private_blocks_count}")
        # --- Component References ---
        self.featurizer = self.model[0]
        self.classifier_head = self.model[1]
        
        self.captured_features = {}
        self.shared_adapter_indices = []  

        # Hook for the private trunk (remains the same)
        if self.private_blocks_count > 0:
            last_private_block = self.featurizer.blocks[self.private_blocks_count - 1]
            last_private_block.register_forward_hook(self._create_hook('private'))

        for i in range(self.private_blocks_count, len(self.featurizer.blocks)):
            block = self.featurizer.blocks[i]
            # A robust check to see if the block has any trainable parameters (i.e., an adapter)
            if any(p.requires_grad for p in block.parameters()):
                hook_name = f'shared_block_{i}'
                block.register_forward_hook(self._create_hook(hook_name))
                self.shared_adapter_indices.append(i)
                print(f"[DEBUG] Intermediate feature hook enabled for shared block: {i}")

        # Ensure we always have a hook on the final block for backward compatibility
        if not self.shared_adapter_indices or self.shared_adapter_indices[-1] != len(self.featurizer.blocks) - 1:
            last_block_idx = len(self.featurizer.blocks) - 1
            if any(p.requires_grad for p in self.featurizer.blocks[last_block_idx].parameters()):
                # Only add if it's not already there
                if last_block_idx not in self.shared_adapter_indices:
                    hook_name = f'shared_block_{last_block_idx}'
                    self.featurizer.blocks[last_block_idx].register_forward_hook(self._create_hook(hook_name))
                    self.shared_adapter_indices.append(last_block_idx)
                    self.shared_adapter_indices.sort() # Keep indices in order
        # --- Conditionally build SOTA components ---
        if self.args.do_bottleneck:
            self.bottleneck = Bottleneck(self.featurizer.num_features)
            print("[DEBUG] --> Information Bottleneck: ENABLED")

        num_domains = len(self.args.site_list)
        # Create SHARED discriminator (for PUSH logic)
        if self.args.do_adv_shared:
            feature_dim = self.featurizer.num_features
            if self.args.per_layer_adv:
                print("[DEBUG] --> Per-Layer Shared Adversarial (PUSH): ENABLED")
                # Create a discriminator for each adaptable shared layer
                discriminators = {}
                for layer_idx in self.shared_adapter_indices:
                    discriminators[f'disc_{layer_idx}'] = DomainDiscriminator(feature_dim, num_domains)
                self.shared_domain_discriminators = nn.ModuleDict(discriminators)
            else:
                print("[DEBUG] --> Final-Layer Shared Adversarial (PUSH): ENABLED")
                # Original behavior: a single discriminator
                self.shared_domain_discriminator = DomainDiscriminator(feature_dim, num_domains)
        # Create PRIVATE discriminator (for PULL logic)
        if self.args.do_adv_private:
            self.private_domain_discriminator = DomainDiscriminator(self.featurizer.num_features, num_domains)
            print("[DEBUG] --> Private Adversarial (PULL): ENABLED")
        if self.args.do_ortho:
            print("[DEBUG] --> Orthogonality Loss: ENABLED")
        print("[DEBUG] FedWrapper initialization complete.\n")

    @property
    def param_prefix(self):
        # This is the name of the attribute holding the nn.Sequential model
        return "model"

    def _create_hook(self, name):
        """Creates a hook function that saves the output of a module."""
        def hook(model, input, output):
            # The output of a ViT block is the full sequence, shape [B, 197, 768]
            # We are interested in the CLS token's representation.
            self.captured_features[name] = output[:, 0]
        return hook

    def _register_hooks(self):
        """
        (Re)registers ALL forward hooks to the internal model's layers.
        Crucial for making deepcopied instances functional.
        """
        # Hook for the private trunk
        if self.private_blocks_count > 0 and self.private_blocks_count <= len(self.model[0].blocks):
            private_block = self.model[0].blocks[self.private_blocks_count - 1]
            private_block.register_forward_hook(self._create_hook('private'))

        # Hook for EACH adaptable shared block, using the stored indices
        for i in self.shared_adapter_indices:
            if i < len(self.model[0].blocks):
                hook_name = f'shared_block_{i}'
                block = self.model[0].blocks[i]
                block.register_forward_hook(self._create_hook(hook_name))

    def get_layer_trainable_params(self, layer_idx):
        """
        Gets the state dict for only the trainable parameters of a specific block.
        This is a robust alternative to string matching.
        """
        if not (0 <= layer_idx < len(self.featurizer.blocks)):
            return OrderedDict()

        block = self.featurizer.blocks[layer_idx]
        layer_state = OrderedDict()
        
        # Find parameters within this specific block
        for name, param in block.named_parameters():
            if param.requires_grad:
                # The key needs to be the full path for load_state_dict,
                # which includes the prefix for the wrapped model.
                full_key = f'model.0.blocks.{layer_idx}.{name}' # <-- FIX: Added "model." prefix
                layer_state[full_key] = param.data.clone()

        # +++ START DEBUG INJECTION +++
        # This tripwire checks for the silent failure condition where a layer
        # SHOULD have trainable parameters, but none were found due to a path mismatch.
        if not layer_state and layer_idx in self.shared_adapter_indices:
            print("\n" + "="*80)
            print("[CRITICAL BUG DETECTED] SILENT FAILURE IN Layer-Wise TTA (L-TTA)!")
            print(f"  - FUNCTION: FedWrapper.get_layer_trainable_params in get_network.py")
            print(f"  - REASON: Failed to find any trainable parameters for block {layer_idx}.")
            print(f"  - IMPACT: L-TTA will not adapt this layer at test-time, using the base global")
            print(f"            model instead. This will degrade performance and invalidate results.")
            print(f"  - LIKELY CAUSE: The hardcoded parameter path prefix 'model.0.blocks...' is")
            print(f"                  no longer valid due to a refactor of the FedWrapper model.")
            print("="*80 + "\n")
        # +++ END DEBUG INJECTION +++

        return layer_state

    def forward(self, x):
        # 1. Clear previous features and run the original, proven forward pass
        self.captured_features.clear()
        
        # We only need the features before the final head
        features_pre_head = self.featurizer(x)
        # This single line executes the entire timm ViT forward pass,
        # triggering our hooks and storing the 'private' and 'shared' features.

        private_features = self.captured_features.get('private', None)

        shared_features_dict = {}
        for i in self.shared_adapter_indices:
            key = f'shared_block_{i}'
            if key in self.captured_features:
                shared_features_dict[i] = self.captured_features[key]
        # 2. Optional: Information Bottleneck
        # The bottleneck is now an external operation, not part of the main flow.
        # It processes the captured private features. This is much cleaner.
        if self.args.do_bottleneck and private_features is not None:
             # This bottleneck doesn't feed back into the model, it's for loss calculation only.
             # This is a simplification. A more complex design could modify the features.
             # For now, let's keep it simple and just calculate the loss.
             _ = self.bottleneck(private_features) # This call populates self.bottleneck.kl_loss

        # 3. Final Classification
        final_output = self.classifier_head(features_pre_head)
        # Handle the case where private trunk doesn't exist
        if private_features is None and shared_features_dict:
            # Use a feature map from the earliest shared block as a shape reference
            ref_tensor = shared_features_dict[min(shared_features_dict.keys())]
            private_features = torch.zeros_like(ref_tensor)        
            
        return final_output, private_features, shared_features_dict

    def get_trainable_params(self, include_private=False):
        """
        A unified function to get trainable parameters from the CORE MODEL.
        Discriminator parameters are NEVER returned by this function as they are
        always treated as local, private components.
        """
        trainable_state = OrderedDict()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                is_private = False
                # Check if the parameter belongs to a private block
                if '.blocks.' in name:
                    try:
                        block_num_str = name.split('.blocks.')[1].split('.')[0]
                        block_num = int(block_num_str)
                        if block_num < self.private_blocks_count:
                            is_private = True
                    except (IndexError, ValueError):
                        pass # Not a block parameter

                if is_private and not include_private:
                    continue # Skip private parameters
                
                trainable_state[name] = param.data.clone()
        
        return trainable_state

    def load_params(self, state_dict):
        """
        A unified function to load parameters. Works for both partial and full state dicts.
        """
        # Load params for the main sequential model
        self.model.load_state_dict(state_dict, strict=False)

def my_get_intermediate_layers(
        featurizer,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        return_prefix_tokens: bool = False,
        norm: bool = False,
):
    """
    Extract intermediate layers from a Vision Transformer model
    Adapted From timm's get_intermediate_layers function
    
    Args:
        featurizer: The ViT model to extract features from
        x: Input tensor
        n: Take last n blocks if int, all if None, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        
    Returns:
        List of intermediate features, optionally tuples of (spatial_tokens, prefix_tokens)
    """
    # Helper function to determine which indices to take
    def feature_take_indices(num_blocks, indices):
        if indices is None:
            take_indices = list(range(num_blocks))
            max_index = num_blocks - 1
        elif isinstance(indices, int):
            take_indices = list(range(num_blocks))[-indices:]
            max_index = num_blocks - 1
        else:
            take_indices = [i for i in indices if i < num_blocks]
            max_index = max(take_indices) if take_indices else 0
        return take_indices, max_index
    
    intermediates = []
    take_indices, max_index = feature_take_indices(len(featurizer.blocks), n)
    
    # forward pass
    B, _, height, width = x.shape
    x = featurizer.patch_embed(x)

    # Add the CLS token before positional embedding
    cls_tokens = featurizer.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    
    # Apply positional embedding
    if hasattr(featurizer, '_pos_embed'):
        x = featurizer._pos_embed(x)
    elif hasattr(featurizer, 'pos_embed'):
        x = x + featurizer.pos_embed
        
    if hasattr(featurizer, 'patch_drop'):
        x = featurizer.patch_drop(x)
    
    if hasattr(featurizer, 'norm_pre'):
        x = featurizer.norm_pre(x)
    
    # Process through blocks and collect intermediates
    blocks = featurizer.blocks[:max_index + 1]
    for i, blk in enumerate(blocks):
        x = blk(x)
        if i in take_indices:
            # normalize intermediates with final norm layer if enabled
            intermediates.append(featurizer.norm(x) if norm else x)
    
    # process intermediates
    num_prefix_tokens = getattr(featurizer, 'num_prefix_tokens', 1)  # Default to 1 (cls token)
    
    if return_prefix_tokens:
        # split prefix (e.g. class, distill) and spatial feature tokens
        prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
        spatial_tokens = [y[:, num_prefix_tokens:] for y in intermediates]
        # Return as list of tuples (spatial_tokens, prefix_tokens)
        return list(zip(spatial_tokens, prefix_tokens))
    
    # If not returning prefix tokens, just return intermediates
    return intermediates


def feats_extractor(x, featurizer, avg_tokens=False, num_layers=4):
    """
    Extract features from a featurizer, ensuring output shape is always [batch_size, feature_dim]
    
    Args:
        x: Input tensor
        featurizer: Feature extractor model
        avg_tokens: Whether to average token features
        num_layers: Number of layers to extract from if mul_layers=True
        
    Returns:
        Tensor of shape [batch_size, feature_dim]
    """
    if num_layers > 1:
        try:
            # Attempt to extract intermediate features using timm's get_intermediate_layers
            intermediate_features = featurizer.get_intermediate_layers(
                x,
                n=num_layers,
                return_prefix_tokens=True,
                norm=True
            )
        except:
            # If failed, use our custom implementation
            intermediate_features = my_get_intermediate_layers(
                featurizer,
                x,
                n=num_layers,
                return_prefix_tokens=True,
                norm=True
            )        
        layers = []
        for spatial_tokens, prefix_tokens in intermediate_features:
            all_tokens = torch.cat([prefix_tokens, spatial_tokens], dim=1)
            if avg_tokens:
                # Average token dimension, preserving [batch_size, embed_dim]
                layers.append(all_tokens.mean(dim=1))
            else:
                # Flatten tokens into a single feature vector per sample
                batch_size = all_tokens.size(0)
                layers.append(all_tokens.reshape(batch_size, -1))
                
        # Average across layers
        final_features = torch.stack(layers).mean(dim=0)
    else:
        # Regular feature extraction
        final_features = featurizer(x)
    
    # Ensure output shape is [batch_size, feature_dim]
    batch_size = final_features.size(0)
    final_features = final_features.reshape(batch_size, -1)
    
    return final_features


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
def GetNetwork(args, num_classes, pretrained=True, **kwargs):
    if args.model == 'resnet18':
        model = ResNet.resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512
        
    elif args.model == 'resnet18_rsc':
        model = ResNet.resnet18_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512

    elif args.model == 'resnet50':
        model = ResNet.resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048
        
    elif args.model == 'resnet50_rsc':
        model = ResNet.resnet50_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048

    elif args.model == 'clip':
        model = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True, num_classes=num_classes)
        feature_level = 768
    
    elif args.model == 'clip_feddgmoe':
        try:
            featurizer = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        except Exception as e:
            print(f"Warning loading model: {e}")
            local_model_path = "/scratch/st-msheha02-1/mosama97/huggingface_cache/models--timm--vit_base_patch16_clip_224.laion2b"
            snapshot_dir = os.path.join(local_model_path, 'snapshots')
            snapshot_folder = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])

            # Construct the full path to the model's config and weights files
            config_path = os.path.join(snapshot_folder, 'config.json')
            weights_path = os.path.join(snapshot_folder, 'pytorch_model.bin')

            # Create the model using the local config and then load the local weights
            featurizer = timm.create_model(
                'vit_base_patch16_clip_224.laion2b',
                pretrained=True,
                pretrained_cfg_overlay=dict(
                    file=weights_path, # Tell it where the weights are
                    hf_hub_id=None, # Prevent it from trying to look online
                    hf_hub_filename=None,
                )
            )

            print(f"Successfully loaded model from local path: {weights_path}")
        feature_level = featurizer.num_features
        featurizer.head = Identity()
        for name, param in featurizer.named_parameters():
            param.requires_grad = False
        # expert_rank = args.expert_rank
        inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where=args.where)

        classifier = nn.Linear(feature_level, num_classes)
        model = nn.Sequential(featurizer, classifier)

    # MODIFIED: Replace the old 'clip_moe' with the new FedTuneMoE
    elif args.model == 'clip_moe_style':
        try:
            featurizer = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        except Exception as e:
            print(f"Warning loading model: {e}")
            local_model_path = "/scratch/st-msheha02-1/mosama97/huggingface_cache/models--timm--vit_base_patch16_clip_224.laion2b"
            snapshot_dir = os.path.join(local_model_path, 'snapshots')
            snapshot_folder = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])

            # Construct the full path to the model's config and weights files
            config_path = os.path.join(snapshot_folder, 'config.json')
            weights_path = os.path.join(snapshot_folder, 'pytorch_model.bin')

            # Create the model using the local config and then load the local weights
            featurizer = timm.create_model(
                'vit_base_patch16_clip_224.laion2b',
                pretrained=True,
                pretrained_cfg_overlay=dict(
                    file=weights_path, # Tell it where the weights are
                    hf_hub_id=None, # Prevent it from trying to look online
                    hf_hub_filename=None,
                )
            )

            print(f"Successfully loaded model from local path: {weights_path}")
        feature_level = featurizer.num_features
        featurizer.head = Identity()
        for name, param in featurizer.named_parameters():
            param.requires_grad = False
        # expert_rank = args.expert_rank
        inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where=args.where)
        classifier = nn.Linear(feature_level, num_classes)

        sequential_model = nn.Sequential(featurizer, classifier)
        
        # 2. Wrap the proven model in our new federated container
        model = FedWrapper(sequential_model, args)
    else:
        raise ValueError("The model is not support")

    return model, feature_level
