from network import ResNet
import torch.nn as nn
from utils.lora_util import inject_trainable_moe_kronecker_new
import timm
import torch
from typing import Union, List, Tuple
from torch.autograd import Function  # <--- ADD THIS LINE

from collections import OrderedDict
import os
# +++ START: ADD THIS NEW WRAPPER CLASS +++
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

        # --- Component References ---
        self.featurizer = self.model[0]
        self.classifier_head = self.model[1]
        
        # --- Hook Setup ---
        self.captured_features = {}

        # Identify the exact layers to hook into
        if self.private_blocks_count > 0:
            last_private_block = self.featurizer.blocks[self.private_blocks_count - 1]
            last_private_block.register_forward_hook(self._create_hook('private'))

        # The last shared block is the final block of the featurizer
        last_shared_block = self.featurizer.blocks[-1]
        last_shared_block.register_forward_hook(self._create_hook('shared'))
        
        # --- Conditionally build SOTA components ---
        if self.args.do_bottleneck:
            self.bottleneck = Bottleneck(self.featurizer.num_features)
        
        if self.args.do_adversarial:
            num_domains = len(self.args.site_list)
            self.domain_discriminator = DomainDiscriminator(self.featurizer.num_features, num_domains)

    def _create_hook(self, name):
        """Creates a hook function that saves the output of a module."""
        def hook(model, input, output):
            # The output of a ViT block is the full sequence, shape [B, 197, 768]
            # We are interested in the CLS token's representation.
            self.captured_features[name] = output[:, 0]
        return hook

    def forward(self, x):
        # 1. Clear previous features and run the original, proven forward pass
        self.captured_features.clear()
        
        # We only need the features before the final head
        features_pre_head = self.featurizer(x)
        # This single line executes the entire timm ViT forward pass,
        # triggering our hooks and storing the 'private' and 'shared' features.

        private_features = self.captured_features.get('private', None)
        shared_features = self.captured_features.get('shared', None)

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
        
        if self.args.do_ortho or self.args.do_adversarial:
            # Ensure we don't fail if a hook didn't run (e.g., private_blocks_count=0)
            if private_features is None:
                private_features = torch.zeros_like(shared_features)
            return final_output, private_features, shared_features
        else:
            return final_output

    # --- get_shared_params and load_shared_params remain UNCHANGED ---
    def get_shared_params(self):
        shared_state = OrderedDict()
        shared_blocks = self.featurizer.blocks[self.private_blocks_count:]
        for name, param in shared_blocks.named_parameters():
            if param.requires_grad:
                shared_state[f'0.blocks.{name}'] = param.data.clone()
        for name, param in self.classifier_head.named_parameters():
            shared_state[f'1.{name}'] = param.data.clone()
        if self.args.do_adversarial and hasattr(self, 'domain_discriminator'):
            for name, param in self.domain_discriminator.named_parameters():
                shared_state[f'domain_discriminator.{name}'] = param.data.clone()
        return shared_state

    def load_shared_params(self, shared_state_dict):
        # We need to load into the base model and potentially the discriminator
        self.model.load_state_dict(shared_state_dict, strict=False)
        if self.args.do_adversarial and hasattr(self, 'domain_discriminator'):
            # The keys for the discriminator won't be in model, so load them separately
            # A bit of care is needed to extract only the relevant keys.
            disc_state = {k.replace('domain_discriminator.',''): v for k,v in shared_state_dict.items() if 'domain_discriminator' in k}
            if disc_state:
                 self.domain_discriminator.load_state_dict(disc_state, strict=False)
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
    
    # +++ START OF MODIFIED/NEW CODE +++

    # Add the CLS token before positional embedding
    cls_tokens = featurizer.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # +++ END OF MODIFIED/NEW CODE +++
    
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
        # inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where='every_qkv', expert_rank=expert_rank)
        inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where='every_qkv')

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
        # inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where='every_qkv', expert_rank=expert_rank)
        inject_trainable_moe_kronecker_new(featurizer, r=[1, 2, 4, 8], where='every_qkv')
        classifier = nn.Linear(feature_level, num_classes)

        sequential_model = nn.Sequential(featurizer, classifier)
        
        # 2. Wrap the proven model in our new federated container
        model = FedWrapper(sequential_model, args)
    else:
        raise ValueError("The model is not support")

    return model, feature_level
