# SCOPE: Shared Content Orchestration of Parameter-Efficient Experts for Federated Domain Generalization

## Requirements
- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0

![OVerview](figs/overview.PNG)
Figure 1. SCOPE’s Hierarchical Federation Strategy. Each client’s model is partitioned by layer depth. The Private Trunk (θp) learns domain-specific features (e.g., style, texture) and remains local, while the Shared Trunk (θs) learns domain-invariant semantic features and is the only component aggregated at the server. The feature heatmaps, generated via Grad-CAM, qualitatively illustrate this functional separation

![Main Framework](figs/main_arch.PNG)
Figure 2. SCOPE Framework Overview. Phase A (Left): Three clients train on different domains with private trunks (domain-specific, remain local) and shared trunks (aggregated at server). Server (Center): ELMA performs expert-level matching using Hungarian algorithm; DRIFT aggregates layer-wise statistics for distribution alignment. Phase B (Right): Layer-wise test-time adaptation where each shared layer independently computes similarity, calculates weights, and blends parameters from all trained domains.

![Results](figs/results.PNG)
Table 1. Leave-one-domain-out evaluation results on OfficeHome, PACS, and VLCS, comparing SCOPE against a comprehensive set of baselines. The best federated performance is in bold.

![Ablation](figs/Ablation.PNG)
Figure 2. SCOPE Framework Overview. Phase A (Left): Three clients train on different domains with private trunks (domain-specific, remain local) and shared trunks (aggregated at server). Server (Center): ELMA performs expert-level matching using Hungarian algorithm; DRIFT aggregates layer-wise statistics for distribution alignment. Phase B (Right): Layer-wise test-time adaptation where each shared layer independently computes similarity, calculates weights, and blends parameters from all trained domains.
## Dataset

Firstly create directory for log files and change the dataset path (`pacs_path`, `officehome_path` and `vlcs_path`) and log path (`log_count_path`) in configs/default.py.

Please download the datasets from the official links:

- [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
- [OfficeHome](https://hemanthdv.github.io/officehome-dataset)
- [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)

## Training / Testing

To train using our best variant for office home and test on domain p, you can run the command:
`
python3 algorithms/feddg_moe/train_officehome.py \
    --test_domain p \
    --batch_size 32 --log_similarities \
    --model clip_moe_style \
    --local_epochs 10 \
    --comm 10 \
    --private_blocks_count 4 \
    --do_migs \
    --migs_lambda_mean 5 \
    --migs_lambda_cov 5 \
    --agg_strategy elma \
    --tta_blend_target layer_wise \
    --note "baseline"


## Acknowledgement

Part of our code is borrowed from the following repositories.
- FedDG-GA [https://github.com/MediaBrain-SJTU/FedDG-GA]
- FACT [https://github.com/MediaBrain-SJTU/FACT]
- DomainBed [https://github.com/facebookresearch/DomainBed]
- MoA [https://github.com/cvlab-kaist/MoA]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]

We thank to the authors for releasing their codes. Please also consider citing their works.
