import os
import sys
# Add the parent directories to the Python path to allow for package-like imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import argparse
from network.get_network import GetNetwork, feats_extractor
from torch.utils.tensorboard.writer import SummaryWriter
from data.domain_net_dataset import domain_net_FedDG
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
# Create a new helper function before main()
# Replace the entire old function with this new version.
def generate_expert_representations(model):
    """
    Directly extracts the expert representation vectors from the `sim_matrix`
    parameter of each CosineTopKGate in the model.
    """
    expert_representations = {}
    # Search within the wrapped model for the routers
    for name, module in model.model.named_modules():
        if isinstance(module, CosineTopKGate):
            # The sim_matrix has shape [proj_dim, num_experts].
            # We want each expert's representation, which is a column.
            # Transposing gives us shape [num_experts, proj_dim], which is what we need.
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
    parser.add_argument("--dataset", type=str, default='domainNet', choices=['domainNet'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='clip_moe',
                        choices=['resnet18', 'resnet50', 'clip', 'clip_moe', 'clip_moe_style'], help='model name')
    parser.add_argument("--where", type=str, default='every_qkv',
                        help='Where to inject the MoE layers in the transformer blocks')
    parser.add_argument("--test_domain", type=str, default='c',
                        choices=['c', 'i', 'p', 'q', 'r', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 345 ', type=int, default=345)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', help='test_batch_size', type=int, default=8)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=10)
    parser.add_argument('--comm', help='epochs number', type=int, default=5)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--agg_strategy', type=str, default='fedavg', choices=['fedavg', 'elma'],
                        help='The server-side aggregation strategy to use.')
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
    parser.add_argument('--tta_blend_target', type=str, default='global_private', choices=['public', 'both', 'global_private', 'disentangled', 'layer_wise'],
                        help='Which trunk(s) to adapt at test time. "public": adapt content, use average style. "both": adapt content and style. "global_private": adapt content, use global style. "disentangled": dual-stream adaptation for style and content.')
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
    # +++ START OF NEW CODE +++
    
    # --- Disentanglement Boundary Arguments (Idea A) ---
    parser.add_argument('--private_blocks_count', type=int, default=4, help='Number of transformer blocks to keep private (style).')

# --- Shared Trunk Adversarial (Push Away from Style) ---
    parser.add_argument('--do_adv_shared', action='store_true', help='Enable adversarial loss on SHARED trunk to enforce domain invariance.')
    parser.add_argument('--adv_lambda_shared', type=float, default=0.1, help='Weight for the SHARED trunk adversarial loss.')

    # --- Private Trunk Adversarial (Pull Toward Style) ---
    parser.add_argument('--do_adv_private', action='store_true', help='Enable adversarial loss on PRIVATE trunk to enforce domain specificity.')
    parser.add_argument('--adv_lambda_private', type=float, default=0.1, help='Weight for the PRIVATE trunk adversarial loss.')

    # --- Orthogonality Regularization Arguments (Idea G) ---
    parser.add_argument('--do_ortho', action='store_true', help='Enable orthogonality regularization loss.')
    parser.add_argument('--ortho_lambda', type=float, default=0.05, help='Weight for the orthogonality loss.')

    # --- Information Bottleneck Arguments (Idea H) ---
    parser.add_argument('--do_bottleneck', action='store_true', help='Enable information bottleneck on the private trunk.')
    parser.add_argument('--beta', type=float, default=1e-5, help='Weight for the KL divergence loss of the bottleneck.')
    # --- Federated Contrastive Invariance (FCI) Loss Arguments ---
    parser.add_argument('--do_fci', action='store_true', help='Enable Federated Contrastive Invariance loss.')
    parser.add_argument('--fci_lambda', type=float, default=0.1, help='Weight for the FCI loss.')
    parser.add_argument('--fci_tau', type=float, default=0.07, help='Temperature parameter (tau) for the FCI loss.')
    # +++ END OF NEW CODE +++
    # misc args
    parser.add_argument('--aux_loss_weight', type=float, default=0.01, help='Weight for the MoE auxiliary loss. Set to 0 to disable.')
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
    # +++ START OF NEW CODE +++
    # --- Print the full argument list to the log file and console ---
    log_file.info("----------------- Training Arguments -----------------")
    print("----------------- Training Arguments -----------------")
    for arg, value in sorted(vars(args).items()):
        log_file.info(f"{arg}: {value}")
        print(f"{arg}: {value}")
    log_file.info("----------------------------------------------------")
    print("----------------------------------------------------")
    # +++ END OF NEW CODE +++    
    '''dataset and dataloader'''
    dataobj = domain_net_FedDG(test_domain=args.test_domain, batch_size=args.batch_size, test_batch_size=args.test_batch_size)
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
    # +++ END OF NEW CODE +++
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

    # --- NEW: Conditional Tracker Setup ---
    first_client_model = model_dict[dataobj.train_domain_list[0]]
    shared_adapter_indices = first_client_model.shared_adapter_indices
    
    domain_stats_private = create_tracker(args)
    domain_stats_shared = None # This will hold either a single tracker or a dict

    if args.tta_blend_target == 'layer_wise':
        domain_stats_shared = {
            i: create_tracker(args) for i in shared_adapter_indices
        }
        log_file.info(f"Initialized L-TTA trackers for shared layers: {shared_adapter_indices}")
    else:
        domain_stats_shared = create_tracker(args)
        log_file.info(f"Initialized a single shared tracker for TTA strategy: '{args.tta_blend_target}'")

    # Simplified stats collection function selection
    collect_stats = collect_offline_stats if 'offline_' in args.domain_tracker else collect_online_stats

    # NEW: Initialize global prototypes
    global_prototypes = None
    feature_dim = global_model.model[0].num_features

    FedUpdate(model_dict, global_model)
    best_val = 0.
    for i in range(args.comm + 1):
        # 1. Distribution: Send global shared params to all clients
        if i > 0: # No distribution on the first round
            # include_private_in_dist = (args.tta_blend_target != 'global_private')
            strategies_that_do_not_aggregate_private_trunk = ['global_private', 'layer_wise']
            include_private_in_dist = (args.tta_blend_target not in strategies_that_do_not_aggregate_private_trunk)
            global_params_to_dist = global_model.get_trainable_params(include_private=include_private_in_dist)
            
            for domain_name in dataobj.train_domain_list:
                model_dict[domain_name].load_params(global_params_to_dist)

        # 2. Local Training and Stats Collection
        local_prototypes_list = []
        for domain_id, domain_name in enumerate(dataobj.train_domain_list):
            feddgmoe_site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                      scheduler_dict[domain_name], dataloader_dict[domain_name]['train'], log_ten, metric, domain_id, global_prototypes=global_prototypes)
            
            collect_stats(model_dict[domain_name], dataloader_dict[domain_name]['train'], domain_stats_private, domain_stats_shared, domain_id, args)
            # NEW: After training, compute local prototypes for this client
            print(f"Computing local prototypes for client {domain_name}...")
            local_protos = compute_local_prototypes(
                model_dict[domain_name],
                dataloader_dict[domain_name]['train'], # Use train loader for prototypes
                args.num_classes,
                feature_dim
            )
            local_prototypes_list.append(local_protos)
            # Optional: You can still evaluate the performance of each full client model
            site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')

        # 3. Aggregation: Collect shared params and average them into the global model
        strategies_that_do_not_aggregate_private_trunk = ['global_private', 'layer_wise']
        include_private_in_agg = (args.tta_blend_target not in strategies_that_do_not_aggregate_private_trunk)
        
        if include_private_in_agg:
            log_file.info(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating BOTH private and public trunks.")
        else:
            log_file.info(f"[INFO] TTA Strategy '{args.tta_blend_target}': Aggregating PUBLIC trunk ONLY.")
        
        # The rest of the code is now correct and does not need to be changed.
        client_params_to_agg = [model_dict[name].get_trainable_params(include_private=include_private_in_agg) for name in dataobj.train_domain_list]
        
        client_weights = [weight_dict[name] for name in dataobj.train_domain_list]

        if args.agg_strategy == 'elma':
            # --- ELMA Strategy ---
            log_file.info("Performing Expert-Level Matched Aggregation (ELMA)...")
            # Generate representations for the global model and all clients
            global_reps = generate_expert_representations(global_model)
            client_reps = [generate_expert_representations(model_dict[name]) for name in dataobj.train_domain_list]

            # Call the new aggregation function
            global_avg_shared_params, permutation_maps = ExpertMatchedAvg(client_params_to_agg, client_weights, global_reps, client_reps, args)
        # +++ DEBUG PRINT +++
            # Check the permutation for the first client and a sample layer to see if it's working
            first_client_perms = permutation_maps[0]
            sample_layer = list(first_client_perms.keys())[0] if first_client_perms else None
            if sample_layer:
                perm_string = str(first_client_perms[sample_layer])
                log_file.info(f"[DEBUG ELMA] Round {i}, Client 0, Sample Layer '{sample_layer}' Permutation: {perm_string}")
                # Check for silent failure
                identity_perm = list(range(len(first_client_perms[sample_layer])))
                if list(first_client_perms[sample_layer]) == identity_perm:
                    log_file.warning(f"[WARNING ELMA] Permutation for client 0 is the identity. ELMA may not be effective.")
            # +++ END DEBUG +++    
        else:
            # --- Standard FedAvg ---
            log_file.info("Performing standard FedAvg...")
            global_avg_shared_params = FedAvg(client_params_to_agg, client_weights)

        global_model.load_params(global_avg_shared_params)
        # --- 3b. NEW: Aggregate Prototypes ---
        if args.do_fci and local_prototypes_list:
            print("Aggregating local prototypes into global prototypes...")
            # Stack all local prototype tensors and average them
            global_prototypes = torch.stack(local_prototypes_list).mean(dim=0)
        # 4. Validation of the new global model
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, global_model, dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val += results_dict['acc'] * weight_dict[domain_name]
        
        if fed_val >= best_val:
            # ... (Saving checkpoints logic is fine, but save global_model)
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            log_file.info(f'Global model saved! Best Val Acc: {best_val*100:.2f}%')
            print(f'Global model saved! Best Val Acc: {best_val*100:.2f}%')

        # 5. Test using the new hybrid evaluation function
        FedTune_testsite_eval(i, args.test_domain, args, global_model, dataloader_dict[args.test_domain]['test'], 
                         log_file, log_ten, metric, note='test_domain', model_dict=model_dict, domain_stats_private=domain_stats_private, domain_stats_shared=domain_stats_shared,
                         weight_dict=weight_dict)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()