SAVE_DIR="/projects/p32646/new_model/"
git clone https://github.com/02lyx/E-dpo.git
cd E-dpo
bash env_setup.sh

source ~/.bashrc
# Initialize Conda environment
eval "$(conda shell.bash hook)"
huggingface-cli login --token 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
# Base paths and settings
initial_model="google/gemma-1.1-7b-it"
base_path="./e_dpo_Gemma-1.1-7b-it"
mkdir $base_path
iteration_prefix="Train"
task_pre="math"
task_suf="metamath"
# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5
    my_world_size=8
    conda activate yy
    python xiaojun_E_step_ent_PPO_yt.py --model_name ${model_path} --critic_model_name "google/gemma-2-2b-it" --learning_rate 4e-7 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --task_type "${task_pre}_${task_suf}${split}" --model_path $e_model_dir || exit 1
    wait
    conda activate gen-eval


    infer_model="${e_model_dir}/final_checkpoint"
    prompt_dir=$3
    output_dir=$4

    CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} &
    CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 30 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} &
    wait
    python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.json" --num_datasets $my_world_size
    conda activate anno-train
    python annotate_data/true_label.py --ds_dir "${output_dir}_data.json" --output_dir $model_output
    wait

    conda activate anno-train
    cat <<EOT > dpo_config.yaml
run_name: $iteration
output_dir: $iteration
model_name_or_path: $model_path
ref_model: $model_path
learning_rate: 4.0e-7
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
gradient_accumulation_steps: 2
label_smoothing: 0.1
EOT

    accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py dpo_config.yaml
}


# Main loop for iterations
for i in {1..3}
do
    iteration_name="Gemma-1.1-7b-it_iter${i}_nocot"
    jsonl_input="Yuanxin-Liu/RLHF4MATH_iter${i}_4k"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"
    e_model_dir="${base_path}/e-model-iter-$i"
    if [ "$i" -eq 1 ]; then
        split="[1000:2000]"
    elif [ "$i" -eq 2 ]; then
        split="[21000:22000]"
    else
        split="[23000:24000]"
    fi
    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="./Gemma-1.1-7b-it_iter${previous_iteration}_nocot"
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output
done

find Gemma-1.1-7b-it_iter3_nocot -maxdepth 1 -type f -exec cp {} ${SAVE_DIR} \;