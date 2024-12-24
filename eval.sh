CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --multi_gpu --num_processes 4 \
    --main_process_port 29501 \
    -m lm_eval --model hf \
    --tasks gsm8k \
    --model_args pretrained="ZhangShenao/baseline-gemma-2-2b-it-sft",parallelize=True \
    --batch_size 8 \
    --output_path ./Logs \
    --log_samples \
# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8
# lm_eval --model hf \
# --model_args pretrained=pipeline_test/baseline_sft \
# --tasks gsm8k \
# --device cuda:0 \
# --batch_size 8 \
# --output_path ./Logs \
# --log_samples \