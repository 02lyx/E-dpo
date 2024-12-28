CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
    --main_process_port 29501 \
    -m lm_eval --model hf \
    --tasks gsm8k \
    --model_args pretrained="/projects/p32646/E-dpo/Gemma-2-2b-it_iter3/checkpoint-1824",parallelize=True \
    --batch_size 32 \
    --output_path ./Logs \
    --log_samples \
# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8
# lm_eval --model hf \
# --model_args pretrained="/projects/p32646/E-dpo/Gemma-2-2b-it_iter1/checkpoint-1882" \
# --tasks gsm8k \
# --device cuda:0 \
# --batch_size 32 \
# --output_path ./Logs \
# --log_samples \