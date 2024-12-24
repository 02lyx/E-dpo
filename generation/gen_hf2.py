#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json, os
from parser import extract_answer, strip_string

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})

# Token name: For repo Iterative-DPO
os.environ["HF_TOKEN"] = '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'
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    #stop=["<|user|>"],
    stop=['Question:'],
)

ds = load_dataset(script_args.dataset_name_or_path)['train']

# instruct_prompt = r"Answer the question based on the following example:"
# example1 = r"""Question: Jack is stranded on a desert island. He wants some salt to season his fish. He collects 2 liters of seawater in an old bucket. If the water is 20% salt, how many ml of salt will Jack get when all the water evaporates? Answer: First find how many liters of the seawater are salt: 2 liters * 20% = 0.4 liters Then multiply that amount by 1000 ml/liter to find the number of ml of salt Jack gets: 0.4 liters * 1000 ml/liter = 400 ml."""
# example2 = r"""Question: Samantha’s last name has three fewer letters than Bobbie’s last name. If Bobbie took two letters off her last name, she would have a last name twice the length of Jamie’s. Jamie’s full name is Jamie Grey. How many letters are in Samantha’s last name? Answer: There are 4 letters in Jamie’s last name, so Bobbie’s name is 4*2 +2 = 10 letters long. Samantha’s last name is 3 letters shorter than Bobbie’s, so there are 10 - 3 = 7 letters in Samantha’s last name."""
# few_shot_cot_prompt = instruct_prompt + '\n' + example2 + f'\nQuestion: '  #'\n' + example1

## for the dataset: RLHFlow/test_generation_2k
# def tokenize(sample):
#     if sample['']
#     answer_text = sample['solution'].split('####')[-1].strip()
#     sample["few_shot_cot_question"] = few_shot_cot_prompt + sample['question']
#     sample["answer_text"] = f"The answer is {answer_text}."
#     return sample

instruct_prompt = r"Please reason step by step, and put your final answer within \\boxed{{}}"
example = r"Question: What is the sum of the following infinite geometric series: $\frac{1}{3} - \frac{1}{9} + \frac{1}{27} - \frac{1}{81} + \frac{1}{243} - \frac{1}{729} + \frac{1}{2187} - \cdots$? Answer: To find the sum of an infinite geometric series, we need to determine if the series converges or diverges. In this case, we have a common ratio of $-\frac{1}{3}$. The series will converge if the absolute value of the common ratio is less than 1, and it will diverge otherwise. In this case, since $\left| -\frac{1}{3} \right| = \frac{1}{3} < 1$, the series converges. Next, we can use the formula for the sum of an infinite geometric series to find its value. The formula is given by: $$ S = \frac{a}{1-r} $$ where $S$ is the sum of the series, $a$ is the first term, and $r$ is the common ratio. In this case, the first term $a$ is $\frac{1}{3}$ and the common ratio $r$ is $-\frac{1}{3}$. Plugging these values into the formula, we have: $$ S = \frac{\frac{1}{3}}{1 - \left(-\frac{1}{3}\right)} $$ Simplifying the denominator, we get: $$ S = \frac{\frac{1}{3}}{\frac{4}{3}} $$ Dividing the numerator and denominator, we obtain: $$ S = \frac{1}{4} $$ Therefore, the sum of the given infinite geometric series is $\boxed{\frac{1}{4}}$. The answer is: \frac{1}{4}"
few_shot_cot_prompt = instruct_prompt + '\n' + example + f'\nQuestion: '  

def tokenize(sample):
    sample["prompt"] = few_shot_cot_prompt + sample['question']
    if sample['type'] in ['gsm8k']:
        answer_text = sample['solution'].split('####')[-1].strip()
    elif sample['type'] in ['gpt-3.5-turbo', 'math', 'MATH_Rephrased', 'MATH_FOBAR', 'MATH_SV', 'GSM_Rephrased', 'GSM_SV', 'GSM_FOBAR']:
        answer_text = extract_answer(sample['solution'])
    else:
        answer_text = ""
        print("error: unknown type")
    sample["answer"] = strip_string(answer_text)
    return sample
# ds = ds.map(
#     lambda x: {
#         "prompt": tokenizer.apply_chat_template(x[script_args.dataset_key], tokenize=False, add_generation_prompt=True)
#     }
# )

ds = ds.map(tokenize, num_proc=16)

data_size = len(ds)
one_num_share = int(data_size / script_args.my_world_size)
ds = ds.select(np.arange(script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share))

print([script_args.local_index * one_num_share, (script_args.local_index + 1) * one_num_share])
print(ds, script_args.dataset_name_or_path)
print(ds[0])


prompts = [ds[i]["prompt"] for i in range(len(ds))]
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
used_prompts = []
gathered_data = []
for i, output in enumerate(outputs):
    tmp_data = {"prompt": ds[i]["prompt"], "question": ds[i]["question"], "responses": [out.text for out in output.outputs], "answer": ds[i]["answer"]}
    gathered_data.append(tmp_data)


print("I collect ", len(gathered_data), "samples")

# os.makedirs(script_args.output_dir, exist_ok=True)
with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
    for i in range(len(gathered_data)):
        json.dump(gathered_data[i], f, ensure_ascii=False)
        f.write('\n')