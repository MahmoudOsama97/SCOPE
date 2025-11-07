import os
import sys
# Add the parent directories to the Python path to allow for package-like imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import re
import argparse
from network.get_network import GetNetwork, feats_extractor
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from utils.classification_metric import Classification 
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate, ExpertMatchedAvg, permute_state_dict
from utils.trainval_func import site_evaluation, feddgmoe_site_train, FedTune_testsite_eval, GetFedModel, SaveCheckPoint, print_trainable_parameters, compute_local_prototypes
import torch.nn.functional as F
from tqdm import tqdm
import torch
import copy # <-- Add this import at the top

from utils.domain_stats.online.online_cosine import OnlineCosineTracker
from utils.domain_stats.online.online_gmm import OnlineGMMTracker
from utils.domain_stats.online.online_gaussian import OnlineGaussianTracker

from utils.domain_stats.offline.offline_gmm import OfflineGMMTracker
from utils.domain_stats.offline.offline_mahalanobis import OfflineMahalanobisTracker
from utils.domain_stats.offline.offline_cosine import OfflineCosineTracker, OfflineCosineMuVarTracker, OfflineCosineConcat
from utils.lora_util import CosineTopKGate

def debug_server_side_routers(client_states, client_names, private_blocks_count):
    """
    Prints the raw `sim_matrix` from the router of each client for direct comparison.

    This helps to visually inspect if the learned expert embeddings are diverging.
    """
    if not client_states or len(client_states) < 2:
        print("\n--- DEBUG: Cannot compare server-side routers (need at least 2 clients) ---\n")
        return

    print(f"\n--- DEBUG: Server-Side Router `sim_matrix` Raw Weight Comparison ---")
    
    # --- Find all public router sim_matrix keys from the first client ---
    ref_state = client_states[0]
    public_router_keys = []
    for key in ref_state.keys():
        if ".router.sim_matrix" in key:
            try:
                block_num = int(re.search(r'blocks\.(\d+)', key).group(1))
                if block_num >= private_blocks_count:
                    public_router_keys.append(key)
            except (AttributeError, IndexError, ValueError):
                print(f"[WARNING] Could not determine block number for router key: {key}. Skipping.")
                continue
    
    if not public_router_keys:
        print("\n[WARNING] Could not find any public router `sim_matrix` parameters to compare!\n")
        return

    # --- To avoid spamming the log, let's inspect only the FIRST public router ---
    key_to_inspect = public_router_keys[0]
    print(f"\n--- Inspecting Router Matrix: {key_to_inspect} ---")
    
    # --- 1. Print the raw sim_matrix tensor for each client ---
    for i, name in enumerate(client_names):
        print(f"  Client '{name}':")
        tensor = client_states[i][key_to_inspect]
        # Print the tensor. We can use options to format it nicely.
        torch.set_printoptions(precision=4, sci_mode=False)
        print(tensor)
        print("-" * 20) # Separator

    # --- 2. For a quantitative comparison, also print the L2 distance ---
    print(f"  Pairwise L2 Distance (Delta) from reference client '{client_names[0]}':")
    ref_tensor = client_states[0][key_to_inspect]
    for i in range(1, len(client_names)):
        name = client_names[i]
        comp_tensor = client_states[i][key_to_inspect]
        delta_norm = torch.norm(ref_tensor - comp_tensor).item()
        print(f"    - Delta vs {name:<8}: {delta_norm:.6f}")
    
    print("\n--- End of Server-Side Router Comparison ---\n")
    # Reset print options to default to not affect other parts of the code
    torch.set_printoptions(profile="default") 

def debug_server_side_params(client_states, client_names, private_blocks_count):
    """
    Prints a detailed comparison of expert parameters received at the server side.

    For each expert parameter, it shows:
    1. Key statistics (Norm, Mean, Std) for each client side-by-side.
    2. The L2 norm of the difference between each client's parameter and the first client's.
    """
    if not client_states or len(client_states) < 2:
        print("\n--- DEBUG: Cannot compare server-side params (need at least 2 clients) ---\n")
        return

    print(f"\n--- DEBUG: Server-Side Parameter Comparison (Comparing all to '{client_names[0]}') ---")

    # First, find all the public expert parameter keys from the first client
    ref_state = client_states[0]
    public_expert_keys = []
    for key in ref_state.keys():
        if 'proj_adapter' in key:
            try:
                # Extract block number to ensure it's a public block
                block_num = int(re.search(r'blocks\.(\d+)', key).group(1))
                if block_num >= private_blocks_count:
                    public_expert_keys.append(key)
            except (AttributeError, IndexError, ValueError):
                continue # Skip keys that don't match the block pattern

    if not public_expert_keys:
        print("\n[ERROR] Could not find any public expert parameters to compare!\n")
        return

    # Now, iterate through each parameter and compare across clients
    for key in public_expert_keys:
        print(f"\n--- Parameter: {key} ---")
        
        # --- 1. Print Side-by-Side Statistics ---
        print("  Client Statistics:")
        header = f"    {'Client':<10} | {'L2 Norm':<15} | {'Mean':<15} | {'Std Dev':<15}"
        print(header)
        print("    " + "-" * len(header))
        
        for i, name in enumerate(client_names):
            tensor = client_states[i][key]
            norm = torch.norm(tensor).item()
            mean = tensor.mean().item()
            std = tensor.std().item()
            print(f"    {name:<10} | {norm:<15.6f} | {mean:<15.6f} | {std:<15.6f}")

        # --- 2. Print Pairwise L2 Distance from the reference client ---
        print("\n  Pairwise L2 Distance (Delta) from reference client '{}':".format(client_names[0]))
        ref_tensor = client_states[0][key]
        for i in range(1, len(client_names)): # Start from the second client
            name = client_names[i]
            comp_tensor = client_states[i][key]
            
            # This is the most important metric: the magnitude of the difference vector
            delta_norm = torch.norm(ref_tensor - comp_tensor).item()
            print(f"    - Delta vs {name:<8}: {delta_norm:.6f}")
    
    print("\n--- End of Server-Side Parameter Comparison ---\n")

def debug_expert_representations(model, site_name, stage=""):
    """
    Prints summary statistics (L2 Norm) for each expert in the model's MoE layers.
    Returns a dictionary of the full representations for later comparison.
    """
    print(f"\n--- DEBUG: Expert Representations for site '{site_name}' ({stage}) ---")
    
    # Store representations to return them for calculating the delta
    representations = {}
    
    # Find all the MoE layers in the model
    for name, module in model.model.named_modules():
        if 'K_Linear_MoE_new_' in module.__class__.__name__:
            # Use the helper method to get the [num_experts, D] tensor of weights
            expert_signatures = module.get_expert_signatures()
            representations[name] = expert_signatures.detach().cpu().clone()
            
            print(f"  Layer '{name}':")
            for i in range(expert_signatures.shape[0]):
                expert_vec = expert_signatures[i]
                # The L2 norm is the magnitude of the vector. A good single-number summary.
                norm = torch.norm(expert_vec).item()
                print(f"    - Expert {i}: L2 Norm = {norm:.6f}")
                
    return representations

def extract_expert_weights_as_vectors(model, include_private=False):
    """
    Extracts the flattened, concatenated weights of each expert in each MoE layer.
    (Final, Direct, Non-Parsing Version with Debugging)
    """
    print(f"\n--- Running extract_expert_weights_as_vectors (include_private={include_private}) ---")
    expert_signatures = {}
    private_blocks_count = model.private_blocks_count

    # Iterate through the model's modules to find the MoE adapters directly
    for module_name, module in model.model.named_modules():
        # Check if the module is the specific MoE adapter class we are using
        if 'K_Linear_MoE_new_' in module.__class__.__name__:
            print(f"[DEBUG EXTRACT] Found MoE module: '{module_name}'")
            
            # --- TRUNK AWARENESS LOGIC ---
            try:
                block_num = int(re.search(r'blocks\.(\d+)', module_name).group(1))
                print(f"[DEBUG EXTRACT]   - Parsed block number: {block_num}")

                if block_num < private_blocks_count and not include_private:
                    print(f"[DEBUG EXTRACT]   - SKIPPING (Private Trunk)")
                    continue 
                else:
                    print(f"[DEBUG EXTRACT]   - INCLUDING (Public Trunk or include_private=True)")
            except (AttributeError, IndexError, ValueError):
                print(f"[WARNING EXTRACT] Could not determine block number for module: {module_name}. Skipping.")
                continue
            
            # --- EXTRACT PARAMETERS DIRECTLY ---
            layer_map_key = module_name 
            
            temp_vectors = []
            for i in range(module.n): # module.n is the number of experts
                left_param = module.proj_adapter1_left[i]
                right_param = module.proj_adapter1_right[i]
                
                signature_vector = torch.cat([left_param.data.flatten(), right_param.data.flatten()])
                temp_vectors.append(signature_vector.detach().cpu())
                print(f"[DEBUG EXTRACT]   - Expert {i}: signature vector created with size {signature_vector.shape[0]}")
            
            if not temp_vectors:
                continue

            max_len = max(v.shape[0] for v in temp_vectors)
            print(f"[DEBUG EXTRACT]   - Max signature length for this layer: {max_len}")

            padded_vectors = []
            for i, v in enumerate(temp_vectors):
                padding_needed = max_len - v.shape[0]
                if padding_needed > 0:
                    padded_v = F.pad(v, (0, padding_needed), 'constant', 0)
                    padded_vectors.append(padded_v)
                    print(f"[DEBUG EXTRACT]   - Padded Expert {i} vector to size {padded_v.shape[0]}")
                else:
                    padded_vectors.append(v)
            
            expert_signatures[layer_map_key] = torch.stack(padded_vectors)
            print(f"[DEBUG EXTRACT]   -> Successfully created signature matrix of size: {expert_signatures[layer_map_key].shape}")
            
    print(f"--- Finished extraction. Found signatures for {len(expert_signatures)} public layers. ---\n")
    return expert_signatures

def generate_expert_representations(model, include_private=False):
    """
    Directly extracts the expert representation vectors from the `sim_matrix`
    parameter of each CosineTopKGate in the model.
    
    Args:
        model: The FedWrapper model.
        include_private (bool): If False, only returns representations from the public trunk.
    """
    expert_representations = {}
    private_blocks_count = model.private_blocks_count
    
    # Search within the wrapped model for the routers
    for name, module in model.model.named_modules():
        if isinstance(module, CosineTopKGate):
            is_private = False
            # Check if this router belongs to a private block
            if '.blocks.' in name:
                try:
                    block_num_str = name.split('.blocks.')[1].split('.')[0]
                    block_num = int(block_num_str)
                    if block_num < private_blocks_count:
                        is_private = True
                except (IndexError, ValueError):
                    print(f"[WARNING ELMA] Could not parse block number from router name: {name}")
                    pass # Not a block parameter we need to worry about
                    

            if is_private and not include_private:
                print(f"[INFO ELMA] Skipping private trunk router: {name}")
                continue # Skip this router if it's in the private trunk
                
            # The sim_matrix has shape [proj_dim, num_experts]. Transpose to get [num_experts, proj_dim].
            expert_representations[name] = module.sim_matrix.T.detach().cpu()
            
    return expert_representations

def collect_online_stats(model, dataloader, domain_stats, domain_id, args):
    """
    Collect online statistics for a domain using the provided dataloader.
    Note: online collection saves memory (no need to store all features at once).
    """
    with torch.no_grad():
        for inputs, _, _ in dataloader:
            features = feats_extractor(inputs.cuda(), model, avg_tokens=args.avg_tokens, num_layers=args.num_layers)
            domain_stats.update(features, domain_id)

def collect_migs_stats(model, dataloader):
    """
    Collects layer-wise mean and covariance statistics for the MIGS loss.
    This is independent of any TTA statistics collection.
    """
    model.eval() # Ensure model is in eval mode for stats collection
    
    # Find all adaptable shared layers from the model's hook list
    shared_adapter_indices = model.shared_adapter_indices
    all_shared_features_per_layer = {i: [] for i in shared_adapter_indices}

    with torch.no_grad():
        for inputs, _, _ in dataloader:
            _, _, s_feats_dict = model(inputs.cuda()) # Keep feats on GPU
            for layer_idx, feats in s_feats_dict.items():
                if layer_idx in all_shared_features_per_layer:
                    all_shared_features_per_layer[layer_idx].append(feats) # Append GPU tensor
    
    migs_stats_for_server = {}
    for layer_idx, feat_list in all_shared_features_per_layer.items():
        if feat_list:
            all_feats = torch.cat(feat_list, dim=0) # Concatenate on GPU
            
            mean_val = torch.mean(all_feats, dim=0)
            centered_feats = all_feats - mean_val
            cov_val = (centered_feats.T @ centered_feats) / (all_feats.shape[0] - 1)
            
            # Only move the final, small tensors to CPU for sending to server
            migs_stats_for_server[layer_idx] = {'mean': mean_val.cpu(), 'cov': cov_val.cpu()}
    
    model.train()
    return migs_stats_for_server

def collect_offline_stats(model, dataloader, domain_stats_private, domain_stats_shared, domain_id, args):
    all_private_features = []
    needs_private_stats = (args.tta_blend_target == 'disentangled')

    # Initialize collectors based on the type of domain_stats_shared
    if args.tta_blend_target == 'layer_wise':
        all_shared_features_per_layer = {i: [] for i in domain_stats_shared.keys()}
    else:
        all_shared_features = []

    with torch.no_grad():
        for inputs, _, _ in dataloader:
            _, p_feats, s_feats_dict = model(inputs.cuda())
            
            if needs_private_stats and p_feats is not None:
                all_private_features.append(p_feats)
            
            # Conditionally collect shared features
            if args.tta_blend_target == 'layer_wise':
                for layer_idx, feats in s_feats_dict.items():
                    if layer_idx in all_shared_features_per_layer:
                        all_shared_features_per_layer[layer_idx].append(feats)
            else:
                if s_feats_dict:
                    last_layer_idx = max(s_feats_dict.keys())
                    all_shared_features.append(s_feats_dict[last_layer_idx])

    # Refit private tracker (no change)
    if needs_private_stats and all_private_features:
        domain_stats_private.refit(torch.cat(all_private_features, dim=0), domain_id)
    
    # Conditionally refit shared trackers
    if args.tta_blend_target == 'layer_wise':
        for layer_idx, tracker in domain_stats_shared.items():
            if all_shared_features_per_layer[layer_idx]:
                all_feats = torch.cat(all_shared_features_per_layer[layer_idx], dim=0)
                tracker.refit(all_feats, domain_id)
    else:
        if all_shared_features:
            all_feats = torch.cat(all_shared_features, dim=0)
            domain_stats_shared.refit(all_feats, domain_id)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='clip_moe',
                        choices=['resnet18', 'resnet50', 'clip', 'clip_moe', 'clip_moe_style'], help='model name')
    parser.add_argument("--where", type=str, default='every_qkv',
                        help='Where to inject the MoE layers in the transformer blocks')                        
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 65', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', help='test_batch_size', type=int, default=32)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=10)
    parser.add_argument('--comm', help='epochs number', type=int, default=5)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--agg_strategy', type=str, default='fedavg', choices=['fedavg', 'elma', 'elma_prob_anchor'],
                        help='The server-side aggregation strategy to use.')
    parser.add_argument('--elma_temp', type=float, default=0.5, 
                        help='Temperature for probabilistic anchor selection in ELMA. Low values are greedy, high values are uniform.')

    ## tracker args
    parser.add_argument('--domain_tracker', help='Which domain stats tracker to use', type=str, default='offline_cosine_muvar',
                       choices=['offline_cosine_muvar', 'offline_cosine', 'offline_mahalanobis', 
                               'offline_gmm', 'offline_cosine_concat',
                               'online_cosine', 'online_gaussian', 'online_gmm'])
    parser.add_argument('--avg_tokens', help='Average token features instead of leaving them flat (i.e token_dim * n_tokens)', action='store_true')
    parser.add_argument('--num_layers', help='Number of layers to extract features from', type=int, default=4)
    ### GMM args
    parser.add_argument('--gmm_num_components', help='Number of components for GMM tracker', type=int, default=3)
    parser.add_argument('--gmm_covariance_type', help='Covariance type for GMM', type=str, default='diag', 
                        choices=['full', 'tied', 'diag', 'spherical'])
    ### test time eval args
    parser.add_argument('--tta_blend_target', type=str, default='global_private', 
                        choices=['none', 'public', 'both', 'global_private', 'disentangled', 'layer_wise'],
                        help='Which trunk(s) to adapt at test time. "none" disables TTA and uses the raw global model.')
    parser.add_argument('--batch_agg_type', type=str, default='pre_similarity', 
                      choices=['pre_similarity', 'post_similarity'], 
                      help='How to aggregate batch features: pre_similarity averages features before computing similarities, post_similarity computes similarities per sample then averages')
    parser.add_argument('--log_similarities', help='Log similarity weights during evaluation', action='store_true')
    parser.add_argument('--inv_temp', help='1 / T for offline cos mu var', type=float, default=2.0)
    parser.add_argument('--normalize_features', help='Normalize features by std dev in cosine muvar tracker', action='store_true', default=True)
    parser.add_argument('--no_normalize_features', help='Disable feature normalization by std dev', action='store_false', dest='normalize_features')
    parser.add_argument('--use_prior', help='Use weight_dict as prior distribution for similarities', action='store_true')
    parser.add_argument('--prior_strength', help='Strength of the prior distribution (higher values give more weight to prior)', type=float, default=1.0)
    # parser.add_argument('--expert_rank', type=int, default=8, help='expert rank for the MoE layer')
    # --- Intra-Client Expert Diversity Regularization ---
    parser.add_argument('--do_expert_div', action='store_true', 
                        help='Enable Intra-Client Expert Diversity Regularization loss to encourage expert specialization.')
    parser.add_argument('--expert_div_lambda', type=float, default=0.01, 
                        help='Weight for the expert diversity regularization loss.')
    # --- Mutual Information Gradient Stopper (MIGS) Loss Arguments ---
    parser.add_argument('--do_migs', action='store_true', help='Enable the layer-wise Mutual Information Gradient Stopper (MIGS) loss.')
    parser.add_argument('--migs_lambda_mean', type=float, default=0.1, help='Weight for the MIGS mean alignment loss.')
    parser.add_argument('--migs_lambda_cov', type=float, default=0.1, help='Weight for the MIGS covariance alignment loss.')
    # --- Disentanglement Boundary Arguments (Idea A) ---
    parser.add_argument('--private_blocks_count', type=int, default=4, help='Number of transformer blocks to keep private (style).')

    # --- Shared Trunk Adversarial (Push Away from Style) ---
    parser.add_argument('--do_adv_shared', action='store_true', help='Enable adversarial loss on SHARED trunk to enforce domain invariance.')
    parser.add_argument('--adv_lambda_shared', type=float, default=0.1, help='Weight for the SHARED trunk adversarial loss.')
    parser.add_argument('--per_layer_adv', action='store_true', help='If --do_adv_shared, apply adversarial loss at each shared layer instead of just the final one.')

    # --- Federated Contrastive Invariance (FCI) Loss Arguments ---
    parser.add_argument('--do_fci', action='store_true', help='Enable Federated Contrastive Invariance loss.')
    parser.add_argument('--fci_lambda', type=float, default=0.1, help='Weight for the FCI loss.')
    parser.add_argument('--fci_tau', type=float, default=0.07, help='Temperature parameter (tau) for the FCI loss.')
    parser.add_argument('--per_layer_fci', action='store_true', help='If --do_fci, apply FCI loss at each shared layer instead of just the final one.')

    # --- Private Trunk Adversarial (Pull Toward Style) ---
    parser.add_argument('--do_adv_private', action='store_true', help='Enable adversarial loss on PRIVATE trunk to enforce domain specificity.')
    parser.add_argument('--adv_lambda_private', type=float, default=0.1, help='Weight for the PRIVATE trunk adversarial loss.')

    # --- Orthogonality Regularization Arguments (Idea G) ---
    parser.add_argument('--do_ortho', action='store_true', help='Enable orthogonality regularization loss.')
    parser.add_argument('--ortho_lambda', type=float, default=0.05, help='Weight for the orthogonality loss.')

    # --- Information Bottleneck Arguments (Idea H) ---
    parser.add_argument('--do_bottleneck', action='store_true', help='Enable information bottleneck on the private trunk.')
    parser.add_argument('--beta', type=float, default=1e-5, help='Weight for the KL divergence loss of the bottleneck.')

    # misc args
    parser.add_argument('--aux_loss_weight', type=float, default=1.0, help='Weight for the MoE auxiliary loss. Set to 0 to disable.')
    parser.add_argument('--note', help='note of experimental settings', type=str, default='feddg_moe')
    parser.add_argument('--display', help='display in controller', action='store_true')
    return parser.parse_args()

def main():
    '''log part'''
    file_name = 'fedavg_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    prototype_input = torch.randn(args.batch_size, 3, 224, 224)
    # --- Print the full argument list to the log file and console ---
    log_file.info("----------------- Training Arguments -----------------")
    print("----------------- Training Arguments -----------------")
    for arg, value in sorted(vars(args).items()):
        log_file.info(f"{arg}: {value}")
        print(f"{arg}: {value}")
    log_file.info("----------------------------------------------------")
    print("----------------------------------------------------")
    '''dataset and dataloader'''
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    #manually add a new named arguments to args
    args.site_list = dataobj.train_domain_list
        
    '''model'''
    metric = Classification()
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    # +++ 2. ADD THE FUNCTION CALL HERE +++
    # --- Print parameter count for one of the client models ---
    # We only need to check one, as they all have the same architecture.
    if dataobj.train_domain_list:
        first_client_model = model_dict[dataobj.train_domain_list[0]]
        print_trainable_parameters(first_client_model, log_file)
    log_file.info(f"[INFO] Aggregation strategy set to: {args.agg_strategy.upper()}")
    print(f"[INFO] Aggregation strategy set to: {args.agg_strategy.upper()}")
    # todo get feature level from the GetFedModel
    def create_tracker(args):    
        if args.domain_tracker == 'online_gaussian':
            log_file.info('Using Online Gaussian Tracker')
            print('Using Online Gaussian Tracker')
            return OnlineGaussianTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        elif args.domain_tracker == 'offline_cosine_muvar':
            log_file.info('Using Offline Cosine MuVar Tracker')
            print('Using Offline Cosine MuVar Tracker')
            return OfflineCosineMuVarTracker(args)
        elif args.domain_tracker == 'offline_cosine_concat':
            log_file.info('Using Offline Cosine Concat Tracker')
            print('Using Offline Cosine Concat Tracker')
            return OfflineCosineConcat(args)
        elif args.domain_tracker == 'online_cosine':
            log_file.info('Using Online Cosine Tracker')
            print('Using Online Cosine Tracker')
            return OnlineCosineTracker(num_domains=len(dataobj.train_domain_list))
        elif args.domain_tracker == 'online_gmm':
            log_file.info('Using Online GMM Tracker')
            print('Using Online GMM Tracker')
            return OnlineGMMTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        elif args.domain_tracker == 'offline_gmm':
            log_file.info(f'Using Offline GMM Tracker with {args.gmm_num_components} components and {args.gmm_covariance_type} covariance')
            print(f'Using Offline GMM Tracker with {args.gmm_num_components} components and {args.gmm_covariance_type} covariance')
            return OfflineGMMTracker(args)
        elif args.domain_tracker == 'offline_mahalanobis':
            log_file.info('Using Offline Mahalanobis Tracker')
            print('Using Offline Mahalanobis Tracker')            
            return OfflineMahalanobisTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        elif args.domain_tracker == 'offline_cosine':
            log_file.info('Using Offline Cosine Tracker')
            print('Using Offline Cosine Tracker')
            return OfflineCosineTracker(feature_dim=768, num_domains=len(dataobj.train_domain_list))
        else:
            raise ValueError(f"Unknown domain tracker: {args.domain_tracker}")

    first_client_model = model_dict[dataobj.train_domain_list[0]]
    shared_adapter_indices = first_client_model.shared_adapter_indices
    last_round_permutation_maps = None # <-- ADD THIS: Variable to store final maps        
    domain_stats_private = create_tracker(args)
    domain_stats_shared = None # This will hold either a single tracker or a dict
    # Initialize global stats for MIGS
    global_stats_for_migs = None 
    if args.tta_blend_target == 'layer_wise':
        domain_stats_shared = {
            i: create_tracker(args) for i in shared_adapter_indices
        }
        log_file.info(f"Initialized L-TTA trackers for shared layers: {shared_adapter_indices}")
        print(f"Initialized L-TTA trackers for shared layers: {shared_adapter_indices}")
    else:
        domain_stats_shared = create_tracker(args)
        log_file.info(f"Initialized a single shared tracker for TTA strategy: '{args.tta_blend_target}'")
        print(f"Initialized a single shared tracker for TTA strategy: '{args.tta_blend_target}'")

    # Simplified stats collection function selection
    collect_stats = collect_offline_stats if 'offline_' in args.domain_tracker else collect_online_stats

    global_prototypes = None
    feature_dim = global_model.model[0].num_features
    # FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm + 1):
        local_prototypes_list = []
        all_clients_migs_stats = []
        client_accuracies = [] 
        strategies_that_do_not_aggregate_private_trunk = ['global_private', 'layer_wise']
        include_private_in_agg = (args.tta_blend_target not in strategies_that_do_not_aggregate_private_trunk)
        client_params_to_agg = [model_dict[name].get_trainable_params(include_private=include_private_in_agg) for name in dataobj.train_domain_list]

        for domain_id, domain_name in enumerate(dataobj.train_domain_list):
            # reps_before = debug_expert_representations(model_dict[domain_name], domain_name, stage="Before Local Training")
            client_params_to_agg = [model_dict[name].get_trainable_params(include_private=include_private_in_agg) for name in dataobj.train_domain_list]
            debug_server_side_routers(
                client_params_to_agg,
                dataobj.train_domain_list,
                args.private_blocks_count
            )
            feddgmoe_site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                      scheduler_dict[domain_name], dataloader_dict[domain_name]['train'], log_ten, metric, domain_id, global_prototypes=global_prototypes, global_stats=global_stats_for_migs if args.do_migs else None)

            client_params_to_agg = [model_dict[name].get_trainable_params(include_private=include_private_in_agg) for name in dataobj.train_domain_list]
            debug_server_side_routers(
                client_params_to_agg,
                dataobj.train_domain_list,
                args.private_blocks_count
            )
            # This runs based on the tta_blend_target argument
            collect_stats(model_dict[domain_name], dataloader_dict[domain_name]['train'], 
                          domain_stats_private, domain_stats_shared, domain_id, args)
            
            # --- MIGS STATISTICS COLLECTION (DECOUPLED) ---
            if args.do_migs:
                client_migs_stats = collect_migs_stats(model_dict[domain_name], 
                                                       dataloader_dict[domain_name]['train'])
                all_clients_migs_stats.append(client_migs_stats)

            print(f"Computing local prototypes for client {domain_name}...")
            local_protos = compute_local_prototypes(
                model_dict[domain_name],
                dataloader_dict[domain_name]['train'], # Use train loader for prototypes
                args.num_classes,
                feature_dim
            )
            local_prototypes_list.append(local_protos)            
            # Optional: You can still evaluate the performance of each full client model
            results_dict = site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')
            client_accuracies.append(results_dict['acc'])
        # 3. Aggregation: Collect shared params and average them into the global model
        strategies_that_do_not_aggregate_private_trunk = ['global_private', 'layer_wise']
        include_private_in_agg = (args.tta_blend_target not in strategies_that_do_not_aggregate_private_trunk)
        
        if include_private_in_agg:
            log_file.info(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating BOTH private and public trunks.")
            print(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating BOTH private and public trunks.")
        else:
            log_file.info(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating PUBLIC trunk ONLY.")
            print(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating PUBLIC trunk ONLY.")
        
        # The rest of the code is now correct and does not need to be changed.
        client_params_to_agg = [model_dict[name].get_trainable_params(include_private=include_private_in_agg) for name in dataobj.train_domain_list]
        num_params_to_agg = len(client_params_to_agg[0])
        if num_params_to_agg > 0:
            first_param_key = list(client_params_to_agg[0].keys())[0]
            log_file.info(f"[DEBUG SPLIT] Round {i}: Successfully extracted {num_params_to_agg} public parameter tensors for aggregation. First key: '{first_param_key}'")
        else:
            log_file.warning(f"[CRITICAL WARNING SPLIT] Round {i}: Extracted 0 public parameters for aggregation. The global model will not update. Check 'get_trainable_params' and model naming.")
        client_weights = [weight_dict[name] for name in dataobj.train_domain_list]

        if args.agg_strategy == 'elma_prob_anchor':
            log_file.info("Performing Probabilistic Anchor ELMA...")
            print("Performing Probabilistic Anchor ELMA...")
            # GUARD RAIL 1: Check if the accuracy list is empty.
            if not client_accuracies:
                print("Client accuracy list is empty. Cannot perform anchor selection. ")
            # 1. Select the Anchor Client
            acc_tensor = torch.tensor(client_accuracies)
            # GUARD RAIL 2: Check for NaN values in the accuracy scores.
            if torch.isnan(acc_tensor).any():
                print("NaN detected in client accuracies. Cannot compute a valid probability distribution. ")

            # GUARD RAIL 3: Check for division by zero.
            if args.elma_temp <= 0:
                print(f"ELMA temperature must be positive, but got {args.elma_temp}.")            

            # Apply softmax with temperature
            probabilities = F.softmax(acc_tensor / args.elma_temp, dim=0)
            
            # Sample one client index to be the anchor
            anchor_client_idx = torch.multinomial(probabilities, 1).item()
            anchor_client_name = dataobj.train_domain_list[anchor_client_idx]
            log_file.info(f"  Selected '{anchor_client_name}' as anchor for this round (Accuracy: {client_accuracies[anchor_client_idx]*100:.2f}%)")
            print(f"  Selected '{anchor_client_name}' as anchor for this round (Accuracy: {client_accuracies[anchor_client_idx]*100:.2f}%)")

            # 2. Prepare inputs for ExpertMatchedAvg
            client_reps = [generate_expert_representations(model_dict[name], include_private=False) for name in dataobj.train_domain_list]
            reference_reps = client_reps[anchor_client_idx] # The anchor's reps are the reference
            
            # 3. Call the refactored function
            global_avg_experts, global_avg_general_params, personalized_routers, permutation_maps = ExpertMatchedAvg(
                client_params_to_agg, client_weights, global_reps, client_reps, args
            )
            last_round_permutation_maps = permutation_maps #<-- ADD THIS LINE
            # Check the permutation for the first client and a sample layer to see if it's working
            first_client_perms_dict = permutation_maps[0]
            if first_client_perms_dict:
                sample_layer = list(first_client_perms_dict.keys())[0]
                perm_map = first_client_perms_dict[sample_layer]
                perm_string = str(perm_map)
                log_file.info(f"[DEBUG ELMA] Round {i}, Client 0, Sample Permutation: {perm_string}")
                # Check for the silent failure condition
                identity_perm = list(range(len(perm_map)))
                if list(perm_map) == identity_perm:
                    log_file.warning(f"[WARNING ELMA] Round {i}: Permutation for client 0 is the identity. ELMA may not be effective (experts may be too similar).")
            # --- ELMA Distribution ---
            # a) Load the common parts into the global model template
            global_model.load_params(global_avg_experts)
            global_model.load_params(global_avg_general_params)
            # b) Loop to create and distribute a personalized package for each client
            for domain_name in dataobj.train_domain_list:
                # Temporarily load this client's personalized router to create the full package
                global_model.load_params(personalized_routers[domain_name])
                
                # Extract the complete, personalized shared trunk
                personalized_shared_trunk = global_model.get_trainable_params(include_private=False)
                
                # Distribute the personalized trunk
                model_dict[domain_name].load_params(personalized_shared_trunk)
        elif args.agg_strategy == 'elma':
            # --- ELMA Strategy ---
            log_file.info("Performing Expert-Level Matched Aggregation (ELMA)...")
            print("Performing Expert-Level Matched Aggregation (ELMA)...")
            
            # --- MODIFIED LINES: Only get PUBLIC representations ---
            global_reps = generate_expert_representations(global_model, include_private=False)
            client_reps = [generate_expert_representations(model_dict[name], include_private=False) for name in dataobj.train_domain_list]
            # --- END OF MODIFIED LINES ---

            # 3. Call the refactored function
            global_avg_experts, global_avg_general_params, personalized_routers, permutation_maps = ExpertMatchedAvg(
                client_params_to_agg, client_weights, global_reps, client_reps, args
            )
            last_round_permutation_maps = permutation_maps #<-- ADD THIS LINE
            # Check the permutation for the first client and a sample layer to see if it's working
            first_client_perms_dict = permutation_maps[0]
            if first_client_perms_dict:
                sample_layer = list(first_client_perms_dict.keys())[0]
                perm_map = first_client_perms_dict[sample_layer]
                perm_string = str(perm_map)
                log_file.info(f"[DEBUG ELMA] Round {i}, Client 0, Sample Permutation: {perm_string}")
                # Check for the silent failure condition
                identity_perm = list(range(len(perm_map)))
                if list(perm_map) == identity_perm:
                    log_file.warning(f"[WARNING ELMA] Round {i}: Permutation for client 0 is the identity. ELMA may not be effective (experts may be too similar).")
            # --- ELMA Distribution ---
            # a) Load the common parts into the global model template
            global_model.load_params(global_avg_experts)
            global_model.load_params(global_avg_general_params)
            # b) Loop to create and distribute a personalized package for each client
            for domain_name in dataobj.train_domain_list:
                # Temporarily load this client's personalized router to create the full package
                global_model.load_params(personalized_routers[domain_name])
                
                # Extract the complete, personalized shared trunk
                personalized_shared_trunk = global_model.get_trainable_params(include_private=False)
                
                # Distribute the personalized trunk
                model_dict[domain_name].load_params(personalized_shared_trunk)
            # Check the permutation for the first client and a sample layer to see if it's working
            first_client_perms = permutation_maps[0]
            sample_layer = list(first_client_perms.keys())[0] if first_client_perms else None
            if sample_layer:
                perm_string = str(first_client_perms[sample_layer])
                log_file.info(f"[DEBUG ELMA] Round {i}, Client 0, Sample Layer '{sample_layer}' Permutation: {perm_string}")
                print(f"[DEBUG ELMA] Round {i}, Client 0, Sample Layer '{sample_layer}' Permutation: {perm_string}")
                # Check for silent failure
                identity_perm = list(range(len(first_client_perms[sample_layer])))
                if list(first_client_perms[sample_layer]) == identity_perm:
                    log_file.warning(f"[WARNING ELMA] Permutation for client 0 is the identity. ELMA may not be effective.")
                    print(f"[WARNING ELMA] Permutation for client 0 is the identity. ELMA may not be effective.")
        
        else:
            # --- Standard FedAvg ---
            log_file.info("Performing standard FedAvg...")
            print("Performing standard FedAvg...")
            global_avg_shared_params = FedAvg(client_params_to_agg, client_weights)
            last_round_permutation_maps = None #<-- ADD THIS LINE
                        # --- FedAvg Distribution ---
            # a) Load the aggregated trunk into the global model
            global_model.load_params(global_avg_shared_params)
            
            # b) Distribute the same shared trunk to all clients
            strategies_that_do_not_aggregate_private_trunk = ['global_private', 'layer_wise']
            include_private_in_dist = (args.tta_blend_target not in strategies_that_do_not_aggregate_private_trunk)
            params_to_dist = global_model.get_trainable_params(include_private=include_private_in_dist)
            for domain_name in dataobj.train_domain_list:
                model_dict[domain_name].load_params(params_to_dist)
        # ---> SERVER-SIDE AGGREGATION OF MIGS STATISTICS <---
        if args.do_migs and all_clients_migs_stats:
            print("Aggregating local statistics into global MIGS targets...")
            aggregated_stats = {}
            layer_keys = all_clients_migs_stats[0].keys()
            
            for layer_key in layer_keys:
                # Weighted sum for the mean and covariance
                mean_sum = torch.zeros_like(all_clients_migs_stats[0][layer_key]['mean'])
                cov_sum = torch.zeros_like(all_clients_migs_stats[0][layer_key]['cov'])
                
                for client_idx, client_stats in enumerate(all_clients_migs_stats):
                    client_name = dataobj.train_domain_list[client_idx]
                    weight = weight_dict[client_name]
                    mean_sum += client_stats[layer_key]['mean'] * weight # Already on CPU
                    cov_sum += client_stats[layer_key]['cov'] * weight # Already on CPU
                
                aggregated_stats[layer_key] = {'mean': mean_sum, 'cov': cov_sum}
            
            global_stats_for_migs = aggregated_stats


        # global_model.load_params(global_avg_shared_params)
        if args.do_fci and local_prototypes_list:
            print("Aggregating local prototypes into global prototypes...")
            # Stack all local prototype tensors and average them
            global_prototypes = torch.stack(local_prototypes_list).mean(dim=0)
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val += results_dict['acc'] * weight_dict[domain_name]
        if fed_val >= best_val:
            # ... (Saving checkpoints logic is fine, but save global_model)
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            log_file.info(f'Global model saved! Best Val Acc: {best_val*100:.2f}%')
            print(f'Global model saved! Best Val Acc: {best_val*100:.2f}%')

        # --- 5. Create a temporary, permuted model dictionary for this round's TTA ---
        permuted_model_dict_for_this_round = {}
        print(f"last_round_permutation_maps: {last_round_permutation_maps}")
        if args.agg_strategy in ['elma', 'elma_prob_anchor'] and last_round_permutation_maps is not None:
            log_file.info(f"[DEBUG L-TTA] Round {i}: Creating pre-permuted models for TTA evaluation.")
            # Create a fresh set of aligned models for this round's evaluation
            for client_idx, domain_name in enumerate(dataobj.train_domain_list):
                original_model_state = model_dict[domain_name].state_dict()
                client_permutation_map = last_round_permutation_maps[client_idx]
                
                # Permute the state dict to align it to the global reference
                permuted_state = permute_state_dict(original_model_state, client_permutation_map)
                
                # Load this aligned state into a new model instance to avoid altering the original
                aligned_model, _ = GetNetwork(args, args.num_classes)
                aligned_model.load_state_dict(permuted_state)
                permuted_model_dict_for_this_round[domain_name] = aligned_model.cuda()
        else:
            log_file.info(f"[DEBUG L-TTA] Round {i}: Using original (un-permuted) models for TTA evaluation (Strategy: {args.agg_strategy}).")
            # If not using ELMA, the TTA source is just the original models
            permuted_model_dict_for_this_round = model_dict
        
        # --- 6. Test using the correctly aligned models for this round ---
        FedTune_testsite_eval(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], 
                         log_file, log_ten, metric, note='test_domain', 
                         model_dict=permuted_model_dict_for_this_round, #<-- USE THE PERMUTED DICT
                         domain_stats_private=domain_stats_private, domain_stats_shared=domain_stats_shared,
                         weight_dict=weight_dict)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()