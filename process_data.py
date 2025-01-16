from datasets import load_dataset, Dataset
import random

# Load the original dataset
dataset = load_dataset("openai/gsm8k", "main")  # Adjust the split if necessary

def processing_data(sample):
    new_dict = {
        'question': sample['question'],
        'type': 'gsm8k',
        'solution': sample['answer']
    }
    return new_dict

new_dataset = dataset.map(processing_data, num_proc=16)
new_repo_name = "rs_gsm8k"

# Step 6: Push to Hugging Face Hub
new_dataset.push_to_hub(new_repo_name)
print(f"Dataset pushed to Hugging Face Hub as '{new_repo_name}'.")