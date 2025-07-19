from collections import OrderedDict


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