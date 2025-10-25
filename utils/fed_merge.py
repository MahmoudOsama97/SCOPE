from collections import OrderedDict
# Add new imports at the top of the file
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cosine_similarity
import re
import torch

def permute_state_dict(state_dict, permutation_map, model_prefix=""):
    """
    Permutes the expert weights in a state_dict according to a permutation map.
    (Final, Direct, Non-Parsing Version with Debugging)
    """
    print(f"\n--- Running permute_state_dict ---")
    permuted_dict = state_dict.copy()
    num_params_swapped = 0
    total_params_in_scope = 0

    for module_name, p in permutation_map.items():
        # module_name is the exact prefix, e.g., '0.blocks.4.attn.qkv'
        # full_layer_prefix = module_name
        full_layer_prefix = module_name.removesuffix('.router')
        print(f"[DEBUG PERMUTE] Processing layer. Searching for parameters with prefix: '{full_layer_prefix}'")

        expert_param_keys = [k for k in state_dict.keys() if k.startswith(full_layer_prefix) and 'proj_adapter' in k]
        total_params_in_scope += len(expert_param_keys)
        num_experts = len(p) if 'p' in locals() and p is not None else len(expert_param_keys) // 2
        print(f"[DEBUG PERMUTE]   - Found {len(expert_param_keys)} param tensors for {num_experts} experts (left/right pairs).")
        print(f"[DEBUG PERMUTE]   - total_params_in_scope so far: {total_params_in_scope}")

        if not expert_param_keys:
            print(f"[WARNING PERMUTE] No expert parameters found for layer '{module_name}'. Skipping.")
            continue
            
        temp_layer_params = {}
        for key in expert_param_keys:
            # Match the expert index at the VERY END of the string
            match = re.search(r'\.(\d+)$', key) 
            if match:
                i_orig = int(match.group(1))
                if i_orig < len(p):
                    i_permuted = p[i_orig]
                    
                    if i_orig != i_permuted:
                        orig_suffix = f".{i_orig}"
                        permuted_suffix = f".{i_permuted}"
                        new_key = key[:-len(orig_suffix)] + permuted_suffix
                        
                        temp_layer_params[new_key] = state_dict[key]
                        num_params_swapped += 1
                        print(f"[DEBUG PERMUTE]   - Remapping '{key}' -> '{new_key}'")
                    else:
                        # No change, map key to itself
                        temp_layer_params[key] = state_dict[key]
                else:
                    temp_layer_params[key] = state_dict[key]
        
        permuted_dict.update(temp_layer_params)

# --- NEW: Calculate total number of individual weights being permuted ---
    total_individual_weights = 0
    if total_params_in_scope > 0:
        # Get the keys of the parameters that were in scope for permutation
        all_expert_keys = []
        for module_name in permutation_map.keys():
            keys_in_layer = [k for k in state_dict.keys() if k.startswith(module_name) and 'proj_adapter' in k]
            all_expert_keys.extend(keys_in_layer)
        # Sum up the number of elements in each of these tensors
        total_individual_weights = sum(state_dict[k].numel() for k in all_expert_keys)

    # Final summary log
    print(f"--- Finished permutation ---")
    if total_params_in_scope > 0:
        if num_params_swapped > 0:
            print(f"[INFO ELMA - PERMUTATION] Successfully remapped {num_params_swapped} / {total_params_in_scope} expert parameters.")
        else:
            print(f"[INFO ELMA - PERMUTATION] All experts were already aligned. No parameters remapped.")
        print(f"[INFO ELMA - SCOPE] The permutation scope covered {total_params_in_scope} tensors, comprising a total of {total_individual_weights:,} individual weights.")
    elif len(permutation_map) > 0:
        print(f"[ERROR ELMA - PERMUTATION] Found 0 total expert parameters across all layers. Check module discovery logic.")
    print("") # Newline for readability

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