# training
base_dir=/home/pyliu/projects/git_pro/QuantizedEmpirical
export OMP_NUM_THREADS=20
set -x

function finetune_multi(){
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=3229 finetune.py \
        --base_model '/mnt/data/pyliu/llama-7b-hf' \
        --data_path $2 \
        --output_dir /mnt/data/pyliu/checkpoint_pyliu/$1 \
        --batch_size 16 \
        --micro_batch_size 2 \
        --num_epochs 3 \
        --learning_rate 3e-4 \
        --cutoff_len 256 \
        --val_set_size 120 \
        --adapter_name lora $3 > logs/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}

# 7B
# finetune_multi alpaca_lora_7B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-7b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-7b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

# 13B
# finetune_multi alpaca_gptqlora_13B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-13b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-13b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

# 30B
# finetune_multi alpaca_gptqlora_30B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gGate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-30b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama-30b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

# 65B
# finetune_multi alpaca_gptqlora_65B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama-65b-hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama65b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2

# 70B llama2
finetune_multi alpaca_gptqlora_70B_4bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama2/llama2_70B_hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama2-70b-4bit-formulate"\ --use_gradient_checkpointing\ --bits=4

finetune_multi alpaca_gptqlora_70B_2bit $base_dir/alpaca_data_cleaned.json --adapter_name=gptqlora\ --target_modules="['q_proj','k_proj','v_proj','o_proj','up_proj','gate_proj','down_proj']"\ --base_model=/mnt/data/pyliu/llama2/llama2_70B_hf\ --quant_checkpoint="/mnt/data/pyliu/gptq_checkpoints/llama2-70b-2bit-formulate"\ --use_gradient_checkpointing\ --bits=2