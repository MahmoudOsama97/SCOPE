from collections import OrderedDict
# Add new imports at the top of the file
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
import re
import torch

def permute_state_dict(state_dict, permutation_map, model_prefix=""):
    """
    Permutes the expert weights AND router embeddings in a state_dict 
    according to a permutation map. (MODIFIED)
    """
    permuted_dict = state_dict.copy()

    for layer_name, p in permutation_map.items():
        # The layer_name from the map is the prefix for the router, e.g., 'model.0.blocks.4.attn.qkv.router'
        
        # --- 1. Permute the Router's sim_matrix ---
        router_key = f"{layer_name}.sim_matrix"
        if router_key in state_dict:
            # sim_matrix shape is [proj_dim, num_experts]. We permute the columns.
            permuted_dict[router_key] = state_dict[router_key][:, p]

        # --- 2. Permute the Expert Weights (left and right adapters) ---
        # The expert prefix is the layer_name without '.router'
        expert_prefix = layer_name.rsplit('.', 1)[0]
        
        # Find all parameter keys belonging to this layer's experts
        expert_param_keys = [k for k in state_dict.keys() if k.startswith(expert_prefix) and 'proj_adapter' in k]
        
        for key in expert_param_keys:
            # The original expert index is the number after the last dot
            base_key, original_expert_index_str = key.rsplit('.', 1)
            original_expert_index = int(original_expert_index_str)
            
            # Find the new index from the permutation map
            if original_expert_index < len(p):
                permuted_expert_index = p[original_expert_index]
                
                # Create the new key for the permuted parameter
                new_key = f"{base_key}.{permuted_expert_index}"
                permuted_dict[new_key] = state_dict[key]

    return permuted_dict

def ExpertMatchedAvg(client_states, client_weights, reference_reps, client_reps, args):
    """
    Performs Expert-Level Matched Aggregation (ELMA) by aligning each client
    to a common set of reference expert representations.
    """
    permuted_client_states = []
    all_permutation_maps = []
    
    for client_idx, client_state in enumerate(client_states):
        client_rep_to_match = client_reps[client_idx]
        print(f"[INFO ELMA] Processing Client client_rep_to_match {client_idx} for ELMA.")
        print(f"[DEBUG ELMA] Client {client_idx} layers available for matching: {list(client_rep_to_match.keys())}")
        permutation_map = {}
        
        for layer_name, ref_expert_reps in reference_reps.items():
            if layer_name not in client_rep_to_match:
                print(f"[WARNING ELMA] Layer '{layer_name}' not found in client {client_idx} representations. Skipping.")
                continue
            
            client_expert_reps = client_rep_to_match[layer_name]
            
            num_experts = ref_expert_reps.shape[0]
            cost_matrix = torch.zeros((num_experts, num_experts))
            for i in range(num_experts):
                for j in range(num_experts):
                    cost_matrix[i, j] = 1 - cosine_similarity(ref_expert_reps[i].unsqueeze(0), client_expert_reps[j].unsqueeze(0))
            print(f"[DEBUG ELMA] Cost matrix for layer '{layer_name}' (Client {client_idx}):\n{cost_matrix.numpy()}")

            row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
            print(f"[DEBUG ELMA] Optimal assignment for layer '{layer_name}' (Client {client_idx}): {list(zip(row_ind, col_ind))}")
            permutation_map[layer_name] = col_ind

        all_permutation_maps.append(permutation_map)
        
        permuted_state = permute_state_dict(client_state, permutation_map)
        permuted_client_states.append(permuted_state)
        
    return FedAvg(permuted_client_states, client_weights), all_permutation_maps



def Dict_weight(dict_in, weight_in):
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in
    
def Dict_Add(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Cal_Weight_Dict(dataset_dict, site_list=None):
    if site_list is None:
        site_list = list(dataset_dict.keys())
    weight_dict = {}
    total_len = 0
    for site_name in site_list:
        total_len += len(dataset_dict[site_name]['test'])
    for site_name in site_list:
        site_len = len(dataset_dict[site_name]['test'])
        weight_dict[site_name] = site_len/total_len
    return weight_dict

# +++ START: REPLACE THE OLD FedAvg WITH THIS +++
def FedAvg(client_states: list, client_weights: list):
    """
    Performs Federated Averaging on a list of state dictionaries.

    Args:
        client_states (list): A list of model state_dicts to be averaged.
        client_weights (list): A list of weights corresponding to each state_dict.

    Returns:
        OrderedDict: The globally averaged state_dict.
    """
    # Ensure the inputs are valid
    if not client_states or not client_weights or len(client_states) != len(client_weights):
        raise ValueError("Inputs to FedAvg are invalid.")

    # Get the keys from the first state dictionary
    keys = client_states[0].keys()
    avg_state = OrderedDict()

    # Iterate through each parameter key
    for key in keys:
        # Calculate the weighted sum for the current parameter
        weighted_sum = sum(state[key] * weight for state, weight in zip(client_states, client_weights))
        avg_state[key] = weighted_sum

    return avg_state

def FedUpdate(model_dict, global_model):
    global_model_parameters = global_model.state_dict()
    for site_name in model_dict.keys():
        model_dict[site_name].load_state_dict(global_model_parameters)
    return None





def MomentumUpdate(model, teacher, alpha=0.99):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k,v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1-alpha)*model_dict[k]
    teacher.load_state_dict(teacher_dict)