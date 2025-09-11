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


def feddgmoe_site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        feddgmoe_epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id)

# Replace the old feddgmoe_epoch_site_train with this new version
def feddgmoe_epoch_site_train(epochs, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, domain_id):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, _ = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()        
        optimizer.zero_grad()
        
        # --- Flexible model output and loss calculation ---
        model_output = model(imgs)
        if isinstance(model_output, tuple):
            output, p_feats, s_feats = model_output
        else:
            output = model_output

        # 1. Standard Classification Loss
        loss = F.cross_entropy(output, labels)
        
        # 2. MoE Auxiliary Loss (Regulated)
        if args.aux_loss_weight > 0:
            loss_aux = 0
            for n, m in model.named_modules():
                if hasattr(m, 'aux_loss'):
                    # Check for stability before adding
                    if not torch.isnan(m.aux_loss) and not torch.isinf(m.aux_loss):
                        loss_aux += m.aux_loss
            
            # Add the weighted, stable aux loss
            loss += args.aux_loss_weight * loss_aux
        # +++ END OF MODIFIED AUX LOSS LOGIC +++

        # 3. Conditional Adversarial Loss (Idea F)
        if args.do_adversarial:
            # +++ START OF FIX +++
            # Create the correct labels on the fly using the local domain_id
            batch_size = s_feats.shape[0]
            correct_domain_labels = torch.full((batch_size,), fill_value=domain_id, dtype=torch.long, device=s_feats.device)
            # +++ END OF FIX +++

            reversed_shared_features = GradientReversalLayerFn.apply(s_feats, args.adv_lambda)
            domain_pred = model.domain_discriminator(reversed_shared_features)
            
            # Use the correct labels for the loss
            loss_dom = F.cross_entropy(domain_pred, correct_domain_labels)
            loss += loss_dom

        # 4. Conditional Orthogonality Loss (Idea G)
        if args.do_ortho:
            p_feats = F.normalize(p_feats, p=2, dim=1)
            s_feats = F.normalize(s_feats, p=2, dim=1)
            ortho_loss = torch.mean((p_feats * s_feats).sum(dim=1).pow(2))
            loss += args.ortho_lambda * ortho_loss

        # 5. Conditional Bottleneck Loss (Idea H)
        if args.do_bottleneck and hasattr(model, 'bottleneck'):
            loss += args.beta * model.bottleneck.kl_loss
            
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
    elif args.dataset == 'terrainc':
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


def FedTune_testsite_eval(epochs, site_name, args, global_model, dataloader, log_file, log_ten, metric, note='after_fed', model_dict=None, domain_stats=None, weight_dict=None):
    """
    Test-time evaluation for Fed-Tune.
    Temporarily modifies the global model's private trunk for each batch and
    restores it, avoiding deepcopy to preserve hooks.
    """
    global_model.eval()
    
    # --- Get the private blocks from the global model once ---
    private_blocks_count = global_model.private_blocks_count
    private_blocks_module = global_model.model[0].blocks[:private_blocks_count]

    with torch.no_grad():
        for imgs, labels, domain_labels, in tqdm(dataloader):
            imgs = imgs.cuda()
            
            # --- Save the original state of the private blocks ---
            original_private_state = {k: v.clone() for k, v in private_blocks_module.state_dict().items()}

            try:
                # 1. Get similarities using the unmodified global model
                intermediate_features = feats_extractor(imgs, global_model.model[0], avg_tokens=args.avg_tokens, num_layers=args.num_layers)
                similarities = domain_stats.get_domain_weights(intermediate_features)
                batch_similarities = F.softmax(torch.mean(similarities, dim=0) * args.inv_temp, dim=0)

                # 2. Dynamically build and load the new private trunk state
                avg_private_trunk_state = FedAvg(
                    [model_dict[name].model[0].blocks[:private_blocks_count].state_dict() for name in args.site_list],
                    batch_similarities.tolist()
                )
                private_blocks_module.load_state_dict(avg_private_trunk_state)

                # 3. Predict with the temporarily modified global_model (which still has hooks)
                model_output = global_model(imgs)
                if isinstance(model_output, tuple):
                    out = model_output[0]
                else:
                    out = model_output
                
                metric.update(out, labels)

            finally:
                # --- CRUCIAL: Restore the original private blocks state ---
                private_blocks_module.load_state_dict(original_private_state)
            
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


