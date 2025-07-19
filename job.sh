
domains=(
  "p"
  "a"
  "c"
  "r"
)

for domain in "${domains[@]}"; do
    # Run your program
    python3 algorithms/feddg_moe/train_officehome.py \
        --test_domain $domain \
        --batch_size 32 \
        --model clip_moe_style \
        --local_epochs 10 \
        --comm 5


done