source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


# Base paths and settings
initial_model="ZhangShenao/baseline-gemma-2-2b-it-sft"
base_path="./iter_dpo_testGemma-2-2b-it"
mkdir $base_path
iteration_prefix="Test"

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    conda init
    conda activate gen-eval
    #bash generation/register_server.sh $model_path
    #sleep 140
    #python generation/gen_hf2.py --model_name_or_path $model_path dataset_name_or_path $jsonl_input 
    #pkill -f "python -m vllm.entrypoints.api_server"
    my_world_size=4
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 4 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 5 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 6 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 7 --my_world_size ${my_world_size} &
    wait
    python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.json" --num_datasets 4
    # accelerate launch annotate_data/get_rewards.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output
    conda init
    conda activate anno-train
    python annotate_data/true_label.py --ds_dir "${output_dir}_data.json" --output_dir $model_output
    # conda activate rlhflow
    cat <<EOT > dpo_config.yaml
run_name: $iteration
output_dir: $iteration
model_name_or_path: $model_path
ref_model: $model_path
learning_rate: 5.0e-7
num_train_epochs: 2
logging_steps: 2
gradient_checkpointing: true
do_train: true
do_eval: true
eval_steps: 10000
choose_type: max_min
train_dir: $model_output
eval_dir: $model_output
loss_type: sigmoid
lr_scheduler_type: cosine
max_length: 2048
max_prompt_length: 1000
eval_strategy: steps
bf16: true
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
report_to: wandb
label_smoothing: 0.1
EOT

    accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py dpo_config.yaml
}


# Main loop for iterations
for i in {1..3}
do
    iteration_name="testGemma-2-2b-it_iter${i}"
    jsonl_input="Yuanxin-Liu/Test-Dataset"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="./testGemma-2-2b-it_iter${previous_iteration}"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output
done
