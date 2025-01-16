source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

export HF_HOME="/projects/p32646/hf_cache"
huggingface-cli login --token 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

# Base paths and settings
initial_model="google/gemma-1.1-7b-it"
base_path="./iter_dpo_Gemma-1.1-7b-it"
mkdir $base_path
iteration_prefix="Train"

# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5

    conda activate gen-eval

    my_world_size=4
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    # CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} &
    # CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} &
    # wait
    # python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.json" --num_datasets ${my_world_size}
    # accelerate launch annotate_data/get_rewards.py --dataset_name_or_path "${output_dir}_data.jsonl" --output_dir $model_output

    conda activate yy
    
    python annotate_data/rs_select_data.py --ds_dir "${output_dir}_data.json" --output_dir $model_output --repo_id gemma-1.1-7b-rs-gsm8k
    wait
    python inference_xiaojun.py --model_path "${infer_model}" --dataset_path "Yuanxin-Liu/gemma-1.1-7b-rs-gsm8k" --save_prefix "./rs_model" --sft_data_type zq_raw --train_step 7500  --task_type  "math_gsm" --learning_rate "5.0e-5"|| exit 1
}


# Main loop for iterations
for i in {1..1}
do
    iteration_name="Gemma-1.1-7b-it_iter${i}"
    jsonl_input="Yuanxin-Liu/rs_gsm8k"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="./Gemma-1.1-7b-it_iter${previous_iteration}"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output
done
