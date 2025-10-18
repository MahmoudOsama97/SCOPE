from collections import OrderedDict
# Add new imports at the top of the file
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
import re
import torch

def permute_state_dict(state_dict, permutation_map):
    """
    Permutes the expert weights in a state_dict according to a permutation map.
    """
    permuted_dict = state_dict.copy()
    
    # +++ START DEBUG INJECTION +++
    # This tripwire checks if the function actually does any work.
    # If the regex fails, this counter will remain 0.
    num_params_swapped = 0
    # Also, check if the permutation map is non-trivial. A trivial map isn't a bug.
    is_map_non_trivial = any(
        any(i != p_i for i, p_i in enumerate(p))
        for p in permutation_map.values()
    )
    # +++ END DEBUG INJECTION +++

    for layer_name, p in permutation_map.items():
        # p is the permutation for this layer, e.g., [0, 2, 1, 3]
        for i_orig, i_permuted in enumerate(p):
            if i_orig == i_permuted:
                continue # No change needed
                
            # Regex to find the expert weights for the original index
            # This matches keys like '...proj_adapter1_left.0.weight'
            orig_pattern_left = re.compile(f'({re.escape(layer_name)}\.proj_adapter1_left\.{i_orig}\..*)')
            orig_pattern_right = re.compile(f'({re.escape(layer_name)}\.proj_adapter1_right\.{i_orig}\..*)')
            
            # Keys for the target permuted index
            permuted_key_left_str = f'{layer_name}.proj_adapter1_left.{i_permuted}.'
            permuted_key_right_str = f'{layer_name}.proj_adapter1_right.{i_permuted}.'
            
            # Find and swap weights
            for key in list(permuted_dict.keys()):
                match_left = orig_pattern_left.match(key)
                if match_left:
                    suffix = key[match_left.end(1)-len(str(i_orig)):]
                    new_key = permuted_key_left_str + suffix
                    permuted_dict[new_key] = state_dict[key]
                    # +++ START DEBUG INJECTION +++
                    num_params_swapped += 1
                    # +++ END DEBUG INJECTION +++
                match_right = orig_pattern_right.match(key)
                if match_right:
                    suffix = key[match_right.end(1)-len(str(i_orig)):]
                    new_key = permuted_key_right_str + suffix
                    permuted_dict[new_key] = state_dict[key]
                    # +++ START DEBUG INJECTION +++
                    num_params_swapped += 1
                    # +++ END DEBUG INJECTION +++
    # +++ START DEBUG INJECTION +++
    # The CRITICAL check: If the permutation map required changes, but we swapped nothing,
    # it means the regex failed to find the parameter names.
    if is_map_non_trivial and num_params_swapped == 0:
        print("\n" + "="*80)
        print("[CRITICAL BUG DETECTED] SILENT FAILURE IN ELMA AGGREGATION!")
        print("  - FUNCTION: permute_state_dict in fed_merge.py")
        print("  - REASON: The model's expert parameters could not be found via regex.")
        print("  - IMPACT: ELMA is NOT working. Experts are being averaged incorrectly,")
        print("            which will degrade model performance and invalidate results.")
        print("  - LIKELY CAUSE: The internal naming of the MoE layers in 'lora_util.py'")
        print("                  has changed, breaking the hardcoded regex patterns here.")
        print("="*80 + "\n")
    # +++ END DEBUG INJECTION +++
    return permuted_dict

def ExpertMatchedAvg(client_states, client_weights, global_reps, client_reps, args):
    """
    Performs Expert-Level Matched Aggregation (ELMA).
    """
    permuted_client_states = []
    all_permutation_maps = [] # <--- New list to store maps    
    # Iterate through each client
    for client_idx, client_state in enumerate(client_states):
        client_rep = client_reps[client_idx]
        permutation_map = {}
        
        # For each MoE layer that has a representation
        for layer_name, global_expert_reps in global_reps.items():
            if layer_name not in client_rep:
                continue
            
            client_expert_reps = client_rep[layer_name]
            
            # Create a cost matrix based on cosine distance (1 - similarity)
            num_experts = global_expert_reps.shape[0]
            cost_matrix = torch.zeros((num_experts, num_experts))
            for i in range(num_experts):
                for j in range(num_experts):
                    cost_matrix[i, j] = 1 - cosine_similarity(global_expert_reps[i].unsqueeze(0), client_expert_reps[j].unsqueeze(0))
            
            # Solve the assignment problem to find the optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
            permutation_map[layer_name] = col_ind # col_ind[i] is the client expert that matches global expert i
        all_permutation_maps.append(permutation_map) # Store the map
        # Permute the client's state_dict according to the map
        permuted_state = permute_state_dict(client_state, permutation_map)
        permuted_client_states.append(permuted_state)
        
    # Perform standard FedAvg on the now-aligned states
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