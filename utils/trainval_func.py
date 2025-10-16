import os
from network.get_network import GetNetwork
import torch
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
from network.get_network import feats_extractor
import copy
from utils.fed_merge import FedAvg
# --- START OF MODIFIED LINE ---
# Import the GRL class from where it is defined
from network.get_network import GetNetwork, feats_extractor, GradientReversalLayerFn
# --- END OF MODIFIED LINE ---

def compute_local_prototypes(model, dataloader, num_classes, feature_dim):
    """
    Computes local class prototypes by averaging features for each class.
    """
    model.eval()
    all_features = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.cuda()
            _, _, s_feats_dict = model(imgs)
            
            # Use features from the final shared block as the representation
            if not s_feats_dict: continue
            last_layer_idx = max(s_feats_dict.keys())
            s_feats = s_feats_dict[last_layer_idx]

            for i in range(len(labels)):
                all_features[labels[i]].append(s_feats[i])

    prototypes = torch.zeros(num_classes, feature_dim).cuda()
    for i in range(num_classes):
        if all_features[i]:
            class_features = torch.stack(all_features[i], dim=0)
            prototypes[i] = torch.mean(class_features, dim=0)
            
    model.train()
    return prototypes.detach().cpu() # Return on CPU to simulate sending to server

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def epoch_site_train(epochs, site_name, model, optimzier, scheduler, dataloader, log_ten, metric):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        output = model(imgs)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimzier.step()
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()
    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, dataloader, log_ten, metric)

def site_evaluation(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            
            # +++ START OF FIX +++
            # Handle conditional tuple output from the model
            model_output = model(imgs)
            if isinstance(model_output, tuple):
                output = model_output[0]  # For evaluation, we only need the main logits
            else:
                output = model_output
            # +++ END OF FIX +++

            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
    print(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

    return results_dict

def site_evaluation_class_level(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_class_acc', results_dict['class_level_acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}% | C Acc: {results_dict["class_level_acc"]*100:.2f}%')
    print(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}% | C Acc: {results_dict["class_level_acc"]*100:.2f}%')
    return results_dict

def site_only_evaluation(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict


def feddgmoe_site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id, global_prototypes=None):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        feddgmoe_epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id, global_prototypes)

# Replace the old feddgmoe_epoch_site_train with this new version
def feddgmoe_epoch_site_train(epochs, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id, global_prototypes=None):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, _ = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()        
        optimizer.zero_grad()
        
        # --- Flexible model output and loss calculation ---
        output, p_feats, s_feats_dict = model(imgs)

        # --- 1. Standard Task Loss & Aux Loss (Unchanged) ---
        loss = F.cross_entropy(output, labels)
        if args.aux_loss_weight > 0:
            loss_aux = 0
            for n, m in model.named_modules():
                if hasattr(m, 'aux_loss'):
                    if not torch.isnan(m.aux_loss) and not torch.isinf(m.aux_loss):
                        loss_aux += m.aux_loss
            loss += args.aux_loss_weight * loss_aux

        # --- 2. Private Trunk "Style Concentration" Loss (Unchanged) ---
        if args.do_adv_private and p_feats is not None and hasattr(model, 'private_domain_discriminator'):
            batch_size = p_feats.shape[0]
            correct_domain_labels = torch.full((batch_size,), fill_value=domain_id, dtype=torch.long, device=p_feats.device)
            domain_pred = model.private_domain_discriminator(p_feats)
            loss_dom_private = F.cross_entropy(domain_pred, correct_domain_labels)
            loss += args.adv_lambda_private * loss_dom_private

        # --- 3. Shared Trunk "Content Purification" Losses ---
        
        # --- 3a. Per-Layer FCI and/or Adversarial Loss ---
        if (args.do_fci and args.per_layer_fci) or (args.do_adv_shared and args.per_layer_adv):
            total_fci_loss = 0
            total_adv_loss = 0
            num_layers_processed = 0

            if args.do_fci and global_prototypes is not None:
                global_prototypes_gpu = global_prototypes.cuda()
                protos_norm = F.normalize(global_prototypes_gpu, p=2, dim=1)

            for layer_idx, s_feats in s_feats_dict.items():
                
                # Apply Per-Layer FCI Loss
                if args.do_fci and args.per_layer_fci and global_prototypes is not None:
                    s_feats_norm = F.normalize(s_feats, p=2, dim=1)
                    sim_matrix = torch.mm(s_feats_norm, protos_norm.t())
                    total_fci_loss += F.cross_entropy(sim_matrix / args.fci_tau, labels)

                # Apply Per-Layer Adversarial Loss
                if args.do_adv_shared and args.per_layer_adv and hasattr(model, 'shared_domain_discriminators'):
                    discriminator = model.shared_domain_discriminators[f'disc_{layer_idx}']
                    reversed_shared_features = GradientReversalLayerFn.apply(s_feats, args.adv_lambda_shared)
                    batch_size = s_feats.shape[0]
                    correct_domain_labels = torch.full((batch_size,), fill_value=domain_id, dtype=torch.long, device=s_feats.device)
                    domain_pred = discriminator(reversed_shared_features)
                    total_adv_loss += F.cross_entropy(domain_pred, correct_domain_labels)
                
                num_layers_processed += 1
            
            if num_layers_processed > 0:
                if total_fci_loss > 0:
                    loss += args.fci_lambda * (total_fci_loss / num_layers_processed)
                if total_adv_loss > 0:
                    loss += total_adv_loss / num_layers_processed

        # --- 3b. Final-Layer FCI and/or Adversarial Loss (Fallback) ---
        else:
            final_s_feats = s_feats_dict[max(s_feats_dict.keys())] if s_feats_dict else None
            if final_s_feats is not None:
                
                # Apply Final-Layer FCI Loss
                if args.do_fci and global_prototypes is not None:
                    global_prototypes_gpu = global_prototypes.cuda()
                    protos_norm = F.normalize(global_prototypes_gpu, p=2, dim=1)
                    s_feats_norm = F.normalize(final_s_feats, p=2, dim=1)
                    sim_matrix = torch.mm(s_feats_norm, protos_norm.t())
                    loss_fci = F.cross_entropy(sim_matrix / args.fci_tau, labels)
                    loss += args.fci_lambda * loss_fci
                
                # Apply Final-Layer Adversarial Loss
                if args.do_adv_shared and hasattr(model, 'shared_domain_discriminator'):
                    reversed_shared_features = GradientReversalLayerFn.apply(final_s_feats, args.adv_lambda_shared)
                    batch_size = final_s_feats.shape[0]
                    correct_domain_labels = torch.full((batch_size,), fill_value=domain_id, dtype=torch.long, device=final_s_feats.device)
                    domain_pred = model.shared_domain_discriminator(reversed_shared_features)
                    loss += F.cross_entropy(domain_pred, correct_domain_labels)

        # --- 4. Other Losses (Ortho, Bottleneck) - Unchanged ---
        final_s_feats_for_other_losses = s_feats_dict[max(s_feats_dict.keys())] if s_feats_dict else None
        if args.do_ortho and p_feats is not None and final_s_feats_for_other_losses is not None:
            p_feats_norm = F.normalize(p_feats, p=2, dim=1)
            s_feats_norm = F.normalize(final_s_feats_for_other_losses, p=2, dim=1)
            ortho_loss = torch.mean((p_feats_norm * s_feats_norm).sum(dim=1).pow(2))
            loss += args.ortho_lambda * ortho_loss

        if args.do_bottleneck and hasattr(model, 'bottleneck') and p_feats is not None:
            _ = model.bottleneck(p_feats) # This call populates the kl_loss
            loss += args.beta * model.bottleneck.kl_loss

        # --- Backward Pass and Optimizer Step (Unchanged) ---
        if torch.isnan(loss):
            print(f"\n[CRITICAL ERROR] Epoch {epochs}, Batch {i}: Total loss is NaN! Aborting backward pass.")
            continue
        
        loss.backward()
        optimizer.step()
        
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs * len(dataloader) + i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()

def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    
    if args.dataset == 'pacs':
        domain_list = pacs_domain_list
    elif args.dataset == 'officehome':
        domain_list = officehome_domain_list
    elif args.dataset == 'domainNet':
        domain_list = domainNet_domain_list
    elif args.dataset == 'TerraIncognita':
        domain_list = terra_incognita_list
    elif args.dataset == 'vlcs':
        domain_list = vlcs_domain_list
        
    for domain_name in domain_list:
        model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = model_dict[domain_name].cuda()
        optimizer_dict[domain_name] = torch.optim.SGD(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4)
        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=int(total_epochs *0.8), gamma=0.1)
        elif args.lr_policy == 'mul_step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(optimizer_dict[domain_name], milestones=[int(total_epochs*0.3), int(total_epochs*0.8)], gamma=0.1)
        elif args.lr_policy == 'exp95':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.95)
        elif args.lr_policy == 'exp98':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.98)
        elif args.lr_policy == 'exp99':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.99)   
        elif args.lr_policy == 'cos':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dict[domain_name], T_max=total_epochs)
    
    return global_model, model_dict, optimizer_dict, scheduler_dict


def FedTune_testsite_eval(epochs, site_name, args, global_model, dataloader, log_file, log_ten, metric, note='after_fed', model_dict=None, domain_stats_private=None, domain_stats_shared=None, weight_dict=None):
    """
    Test-time evaluation with four strategies: 'public', 'both', 'global_private', and 'disentangled'.
    """
    global_model.eval()
    # +++ DEBUG PRINT AT THE START +++
    print(f"\n[INFO TTA] Running Test-Time Adaptation with strategy: '{args.tta_blend_target.upper()}'")
    # +++ END DEBUG +++
    # --- Pre-compute the "neutral average" private trunk state if needed ---
    avg_private_state = None
    if args.tta_blend_target == 'public':
        private_blocks_count = global_model.private_blocks_count
        all_private_states = [model_dict[name].model[0].blocks[:private_blocks_count].state_dict() for name in args.site_list]
        num_clients = len(args.site_list)
        uniform_weights = [1.0 / num_clients] * num_clients
        avg_private_state = FedAvg(all_private_states, uniform_weights)
    # --- MAIN L-TTA LOGIC ---
    if args.tta_blend_target == 'layer_wise':
        domain_stats_per_layer = domain_stats_shared  # Rename for clarity
        hybrid_model = copy.deepcopy(global_model)
        hybrid_model._register_hooks()
        
        with torch.no_grad():
            for imgs, labels, _, in tqdm(dataloader):
                imgs = imgs.cuda()
                
                # 1. Get all intermediate features for the test batch
                _, _, test_s_feats_dict = global_model(imgs)
                
                # Start with a clean copy of the global model's full state dict
                final_blended_state = global_model.state_dict()

                # 2. Loop through each adaptable layer to blend its parameters
                for layer_idx, tracker in domain_stats_per_layer.items():
                    if layer_idx not in test_s_feats_dict: continue # Skip if no features
                    
                    test_feats = test_s_feats_dict[layer_idx]
                    
                    # Calculate similarity weights for THIS LAYER
                    similarities = tracker.get_domain_weights(test_feats)
                    layer_weights = F.softmax(torch.mean(similarities, dim=0) * args.inv_temp, dim=0)
                    
                    if args.log_similarities:
                        print(f"  [DEBUG L-TTA Layer {layer_idx}] Weights: {[f'{w:.2f}' for w in layer_weights]}")

                    # Extract this layer's adapters from each client
                    layer_states = [
                        model_dict[name].get_layer_trainable_params(layer_idx)
                        for name in args.site_list
                    ]
                    
                    # Blend the parameters for this layer
                    if any(layer_states):
                        blended_layer_params = FedAvg(layer_states, layer_weights.tolist())
                        final_blended_state.update(blended_layer_params)
                
                # 3. Load the fully composed state dict
                hybrid_model.load_state_dict(final_blended_state)
                
                out, _, _ = hybrid_model(imgs)
                metric.update(out, labels)

    # --- LOGIC FOR ORIGINAL TTA STRATEGIES (with compatibility fix) ---
    else:
        with torch.no_grad():
            for imgs, labels, domain_labels, in tqdm(dataloader):
                imgs = imgs.cuda()

                hybrid_model = copy.deepcopy(global_model)
                hybrid_model._register_hooks()

                # --- 1. Get Similarity Weights based on the Strategy ---
                if args.tta_blend_target == 'disentangled':
                    # D-TTA: Dual-stream similarity calculation
                    # Get all features, then extract the ones needed
                    _, test_p_feats, test_s_feats_dict = global_model(imgs)
                    
                    # COMPATIBILITY: Extract final layer features for old methods
                    test_s_feats = None
                    if test_s_feats_dict:
                        last_layer_idx = max(test_s_feats_dict.keys())
                        test_s_feats = test_s_feats_dict[last_layer_idx]
                    # A. Style Similarity
                    style_similarities = domain_stats_private.get_domain_weights(test_p_feats)
                    style_weights = F.softmax(torch.mean(style_similarities, dim=0) * args.inv_temp, dim=0)
                    
                    # B. Content Similarity
                    content_similarities = domain_stats_shared.get_domain_weights(test_s_feats)
                    content_weights = F.softmax(torch.mean(content_similarities, dim=0) * args.inv_temp, dim=0)

                    if args.log_similarities:
                        print(f"  [DEBUG TTA] Style Weights:   {[f'{w:.2f}' for w in style_weights]}")
                        print(f"  [DEBUG TTA] Content Weights: {[f'{w:.2f}' for w in content_weights]}")

                else:
                    # Standard TTA: Single-stream similarity calculation for all other strategies
                    _, _, test_s_feats_dict = global_model(imgs) 

                    test_s_feats = None
                    if test_s_feats_dict:
                        last_layer_idx = max(test_s_feats_dict.keys())
                        test_s_feats = test_s_feats_dict[last_layer_idx]
                    similarities = domain_stats_shared.get_domain_weights(test_s_feats)
                    batch_similarities = F.softmax(torch.mean(similarities, dim=0) * args.inv_temp, dim=0)
                    # +++ DEBUG PRINT +++
                    if args.log_similarities:
                        print(f"  [DEBUG TTA] Standard Weights: {[f'{w:.2f}' for w in batch_similarities]}")
                    # +++ END DEBUG +++

                # --- 2. Execute Blending Strategy ---
                if args.tta_blend_target == 'disentangled':
                    # ** OPTION 4: Disentangled Blending **
                    private_blocks_count = hybrid_model.private_blocks_count
                    private_states = [model_dict[name].model[0].blocks[:private_blocks_count].state_dict() for name in args.site_list]
                    blended_private_state = FedAvg(private_states, style_weights.tolist()) # Use style_weights
                    hybrid_model.model[0].blocks[:private_blocks_count].load_state_dict(blended_private_state)
                    
                    # Get ONLY the shared parameters for content blending
                    shared_states = [model_dict[name].get_trainable_params(include_private=False) for name in args.site_list]
                    blended_shared_state = FedAvg(shared_states, content_weights.tolist())
                    hybrid_model.load_params(blended_shared_state)

                else:
                    # --- Logic for the original three strategies ---
                    # Get ONLY the shared parameters for blending
                    shared_states = [model_dict[name].get_trainable_params(include_private=False) for name in args.site_list]
                    blended_shared_state = FedAvg(shared_states, batch_similarities.tolist())
                    hybrid_model.load_params(blended_shared_state)

                    if args.tta_blend_target == 'both':
                        private_blocks_count = hybrid_model.private_blocks_count
                        private_states = [model_dict[name].model[0].blocks[:private_blocks_count].state_dict() for name in args.site_list]
                        blended_private_state = FedAvg(private_states, batch_similarities.tolist())
                        hybrid_model.model[0].blocks[:private_blocks_count].load_state_dict(blended_private_state)
                    
                    elif args.tta_blend_target == 'public':
                        private_blocks_count = hybrid_model.private_blocks_count
                        hybrid_model.model[0].blocks[:private_blocks_count].load_state_dict(avg_private_state)
                    
                    elif args.tta_blend_target == 'global_private':
                        pass

                # 3. Predict with the assembled hybrid model
                model_output = hybrid_model(imgs)
                if isinstance(model_output, tuple):
                    out = model_output[0]
                else:
                    out = model_output
                
                metric.update(out, labels)
            
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
    print(f'{note} Round: {epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

    return results_dict

# +++ END OF NEW FUNCTION +++

def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    check_dict = {'args':args, 'epochs':epochs, 'model':model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)
        
    torch.save(check_dict, os.path.join(path, note+'.pt'))

def feddgmoe_testsite_eval(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed', model_dict=None, domain_stats=None, weight_dict=None):
    """
    Unified test site evaluation function for FedDG-MoE.
    
    Args:
        epochs: Current epoch number
        site_name: Name of the site/domain being evaluated
        args: Arguments object with configuration
        model: The model to evaluate
        dataloader: Test data loader
        log_file: Logger for output
        log_ten: Tensorboard writer
        metric: Metric tracker object
        note: Note for logging
        model_dict: Dictionary of domain-specific models
        domain_stats: Domain statistics tracker
        weight_dict: Dictionary with domain weights based on dataset sizes, used as prior
    """
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in tqdm(dataloader):
            imgs = imgs.cuda()


            # --- MODIFIED LINE ---
            # Call feats_extractor on the full global_model, not a piece of it.
            intermediate_features = feats_extractor(imgs, global_model, avg_tokens=args.avg_tokens, num_layers=args.num_layers)
            # --- END OF MODIFIED LINE ---

            # Calculate domain similarities based on the aggregation type (raw scores - no softmax yet)
            if args.batch_agg_type == "pre_similarity":
                # Average features first, then compute similarities (v2 behavior)
                batch_averaged_features = torch.mean(intermediate_features, dim=0).unsqueeze(0)
                similarities = domain_stats.get_domain_weights(batch_averaged_features)
            else:  # "post_similarity"
                # Compute similarities for each sample, then average (original behavior)
                similarities = domain_stats.get_domain_weights(intermediate_features)
            
            # First average the raw similarities across the batch
            batch_similarities = torch.mean(similarities, dim=0)
            
            # Apply prior distribution if enabled
            if args.use_prior and weight_dict is not None:
                # Convert weight_dict to tensor in the same order as site_list
                prior_weights = torch.tensor([weight_dict[domain] for domain in args.site_list], device=batch_similarities.device)
                # Log probabilities to incorporate as prior (equivalent to multiplying probabilities)
                log_prior = torch.log(prior_weights) * args.prior_strength
                # Add log prior to the scaled similarities
                batch_similarities = batch_similarities * args.inv_temp + log_prior
                # Apply softmax to get final probabilities
                batch_similarities = F.softmax(batch_similarities, dim=0)
            else:
                # Original behavior: just apply temperature scaling and softmax
                batch_similarities = F.softmax(batch_similarities * args.inv_temp, dim=0)
            
            # Optional logging of similarities
            if args.log_similarities:
                log_file.info(f'{note}_{site_name}_batch_similarities: {batch_similarities} (epoch {epochs})')
                print(f'{note}_{site_name}_batch_similarities: {batch_similarities} (epoch {epochs})')
                
            # Update model parameters based on similarities
            for name, param in model.named_parameters():
                if param.requires_grad:                            
                    NewParam = sum(w * model_dict[train_domain].state_dict()[name] for w, train_domain in zip(batch_similarities, args.site_list))
                    param.data.copy_(NewParam)
                    
            out = model(imgs)
            metric.update(out, labels)
            
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
    print(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')
    
    return results_dict




# +++ START OF NEW FUNCTION +++
def print_trainable_parameters(model, log_file=None):
    """
    Calculates and prints the number of total and trainable parameters in a model.
    """
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Prepare the output strings
    header = "----------------- Model Parameters -----------------"
    line1 = f"Total parameters:     {total_params:,}"
    line2 = f"Trainable parameters: {trainable_params:,}"
    
    if total_params > 0:
        percentage = 100 * trainable_params / total_params
        line3 = f"Trainable percentage: {percentage:.2f}%"
    else:
        line3 = "Trainable percentage: N/A"
    footer = "----------------------------------------------------"

    # Print to console
    print(header)
    print(line1)
    print(line2)
    print(line3)
    print(footer)

    # Log to file if a logger is provided
    if log_file:
        log_file.info(header)
        log_file.info(line1)
        log_file.info(line2)
        log_file.info(line3)
        log_file.info(footer)
# +++ END OF NEW FUNCTION +++