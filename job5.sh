#!/bin/bash
#SBATCH --account=st-msheha02-1-gpu       # Replace with your actual account
#SBATCH --time=96:00:00            # Walltime (days-hours:minutes:seconds) - 4 hours
#SBATCH --nodes=1                    # Number of nodes (you want 4 GPUs on 1 node)
#SBATCH --ntasks-per-node=1          # Number of tasks per node (usually 1 for GPU jobs)
#SBATCH --cpus-per-task=16           # CPUs per task (adjust based on your needs)
#SBATCH --mem=127G                   # Memory per node (adjust based on your needs)
#SBATCH --gpus=1         # Request 4 A100 GPUs (40GB VRAM)
#SBATCH --constraint=gpu_mem_32
#SBATCH --job-name=FEDDGST        # Name of the job
#SBATCH --output=output_%j.out      # Output file (stdout) - %j is the job ID
#SBATCH --error=error_%j.err       # Error file (stderr) - %j is the job ID

# Load modules (same as in your interactive session)
module load gcc
module load cuda

# if the $SLURM_TMPDIR is not set, set it to $TMPDIR
if [ -z "$SLURM_TMPDIR" ]; then
    export SLURM_TMPDIR=$TMPDIR
fi

# Export all temp variables
echo "EXPORTED_SLURM_VARS: $SLURM_TMPDIR"

# Move the code to the compute node
cp -r /scratch/st-msheha02-1/mosama97/FedDG-MoE2 $SLURM_TMPDIR
cp -r /scratch/st-msheha02-1/mosama97/datasets/pacs81.zip $SLURM_TMPDIR/FedDG-MoE2/datasets

echo "EXPORTED_SLURM_VARS: $SLURM_TMPDIR"

# Activate your virtual environment
source /scratch/st-msheha02-1/mosama97/myenv/bin/activate
# 1. Set the PyTorch cache directory (for torchvision)
export TORCH_HOME=/scratch/st-msheha02-1/mosama97/torch_cache

# 2. Set the Hugging Face cache directory (for timm)
export HF_HOME=/scratch/st-msheha02-1/mosama97/huggingface_cache

# 3. Set WandB to run in offline mode (for logging)
export WANDB_MODE=offline
# 4. !!! ADD THIS LINE - The most important change !!!
# Force Hugging Face and Timm to work offline and not attempt any network connection
export TRANSFORMERS_OFFLINE=1
cd $SLURM_TMPDIR/FedDG-MoE2/datasets
mv pacs81.zip pacs.zip
unzip pacs.zip
cd $SLURM_TMPDIR/FedDG-MoE2/
echo "EXPORTED_SLURM_VARS: $SLURM_TMPDIR"
domains=(
  "p"
  "a"
  "c"
  "s"
)

for domain in "${domains[@]}"; do
    # Run your program
    python3 algorithms/feddg_moe/train_pacs.py \
        --test_domain $domain \
        --batch_size 32 --log_similarities \
        --model clip_moe_style \
        --local_epochs 10 \
        --comm 10 \
        --private_blocks_count 4 \
        --do_fci \
        --fci_lambda 0.5 \
        --fci_tau 0.07 \
        --note "baseline"
done

python3 analyze_logs.py --output_folder ./exps --note "baseline"

cp -r $SLURM_TMPDIR/FedDG-MoE2/exps/. /scratch/st-msheha02-1/mosama97/output/
echo "----------------------------------------"


echo ""  # Add an empty line for better readability
