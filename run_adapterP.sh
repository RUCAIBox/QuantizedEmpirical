cd peft
pip install -e .

CUDA_VISIBLE_DEVICES=0 python finetune_ds_adapter.py \
  --base_model '/mnt/liupeiyu/llama_checkpoint/llama-7b-hf' \
  --data_path '/home/liupeiyu/QuantizedEmpirical/data/self_instruct.json' \
  --output_dir '/mnt/textbox/tianyu.su/self_instruct_adapterP' \
  --batch_size 16 \
  --micro_batch_size 2 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name bottleneck \
  --use_adapterp > logs/self_instruct_adapterP.log 2>&1 &
