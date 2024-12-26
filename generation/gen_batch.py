#!/usr/bin/env python
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from vllm import LLM, SamplingParams
from tqdm import tqdm
from parser import extract_answer, strip_string  # Ensure these are correctly implemented


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_q_name_or_path: Optional[str] = field(
        default="your_model_q",
        metadata={"help": "The location of the model Q (generation model)"},
    )
    model_p_name_or_path: Optional[str] = field(
        default="your_model_p",
        metadata={"help": "The location of the model P (reference answer model)"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "The location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "The local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="output/",
        metadata={"help": "The location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "The total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "The number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "The random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "The temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "Use beam search for generation"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "The key of the dataset"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "The IDs of the end-of-sentence tokens"},
    )


def main(script_args):
    # Set seeds for reproducibility
    seed = script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load Tokenizers
    tokenizer_q = AutoTokenizer.from_pretrained(script_args.model_q_name_or_path)
    tokenizer_p = AutoTokenizer.from_pretrained(script_args.model_p_name_or_path)

    # Load Model Q for generation
    llm_q = LLM(
        model=script_args.model_q_name_or_path,
        tokenizer=script_args.model_q_name_or_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=seed,
    )

    llm_p = LLM(
        model=script_args.model_p_name_or_path,
        tokenizer=script_args.model_p_name_or_path,
        dtype="bfloat16",
        load_format="auto",
        seed=seed,
    )

    # Define Sampling Parameters for Model Q
    sampling_params_q = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens,
        n=script_args.K,
        stop_token_ids=[tokenizer_q.eos_token_id] + script_args.eos_ids,
        stop=['Question:'],  # Adjust as needed
    )
    sampling_params_p = SamplingParams(
        temperature=script_args.temperature,
        top_p=1.0,
        max_tokens=script_args.max_new_tokens+20,
        n=1,
        stop_token_ids=[tokenizer_q.eos_token_id] + script_args.eos_ids,
        stop=['Question:'],  # Adjust as needed
    )
    # Load and preprocess dataset
    ds = load_dataset(script_args.dataset_name_or_path)['train']

    instruct_prompt = r"Please reason step by step, and put your final answer within \boxed{}"
    example = (
        r"Question: What is the sum of the following infinite geometric series: $\frac{1}{3} - \frac{1}{9} + \frac{1}{27} - \frac{1}{81} + \frac{1}{243} - \frac{1}{729} + \frac{1}{2187} - \cdots$? "
        r"Answer: To find the sum of an infinite geometric series, we need to determine if the series converges or diverges. In this case, we have a common ratio of $-\frac{1}{3}$. The series will converge if the absolute value of the common ratio is less than 1, and it will diverge otherwise. In this case, since $\left| -\frac{1}{3} \right| = \frac{1}{3} < 1$, the series converges. Next, we can use the formula for the sum of an infinite geometric series to find its value. The formula is given by: $$ S = \frac{a}{1-r} $$ where $S$ is the sum of the series, $a$ is the first term, and $r$ is the common ratio. In this case, the first term $a$ is $\frac{1}{3}$ and the common ratio $r$ is $-\frac{1}{3}$. Plugging these values into the formula, we have: $$ S = \frac{\frac{1}{3}}{1 - \left(-\frac{1}{3}\right)} $$ Simplifying the denominator, we get: $$ S = \frac{\frac{1}{3}}{\frac{4}{3}} $$ Dividing the numerator and denominator, we obtain: $$ S = \frac{1}{4} $$ Therefore, the sum of the given infinite geometric series is $\boxed{\frac{1}{4}}$. The answer is: \frac{1}{4}"
    )
    few_shot_cot_prompt = instruct_prompt + '\n' + example + f'\nQuestion: '

    def tokenize(sample):
        sample["prompt"] = few_shot_cot_prompt + sample['question']
        if sample['type'] in ['gsm8k']:
            answer_text = sample['solution'].split('####')[-1].strip()
        elif sample['type'] in [
            'gpt-3.5-turbo',
            'math',
            'MATH_Rephrased',
            'MATH_FOBAR',
            'MATH_SV',
            'GSM_Rephrased',
            'GSM_SV',
            'GSM_FOBAR',
        ]:
            answer_text = extract_answer(sample['solution'])
        else:
            answer_text = ""
            print("error: unknown type")
        sample["answer"] = strip_string(answer_text)
        return sample

    ds = ds.map(tokenize, num_proc=16)

    # Distribute dataset among agents
    data_size = len(ds)
    one_num_share = int(data_size / script_args.my_world_size)
    start_idx = script_args.local_index * one_num_share
    end_idx = (script_args.local_index + 1) * one_num_share
    ds = ds.select(np.arange(start_idx, end_idx))

    print(f"Processing data from index {start_idx} to {end_idx}")
    print(ds, script_args.dataset_name_or_path)
    if len(ds) > 0:
        print(ds[0])

    prompts = [ds[i]["prompt"] for i in range(len(ds))]

    # Generate K responses using Model Q
    print("Generating responses with Model Q...")
    outputs_q = llm_q.generate(prompts, sampling_params=sampling_params_q, use_tqdm=True)
    tokenizer_p.pad_token = tokenizer_p.eos_token

    gathered_data = []

    print("Generating reference answers with Model P...")
    for i in tqdm(range(len(ds)), desc="Generating reference answers"):
        prompt = ds[i]["prompt"]
        question = ds[i]["question"]
        answer = ds[i]["answer"]
        responses = [out.text for out in outputs_q[i].outputs]
        prompts_p = []
        for response in responses:
            # Prepare the input for Model P
            prompt_p = response + ' The answer is'
            prompts_p.append(prompt_p)
        
        outputs_p = llm_p.generate(prompts_p, sampling_params=sampling_params_p)
        new_response = [out.text for out in outputs_p.outputs]
            # new_prompt_toks = tokenizer_p(prompt_p, return_tensors='pt', truncation=True, padding=True)

            # with torch.no_grad():
            #     # Generate reference answer
            #     seq = ref_model.generate(
            #         input_ids=new_prompt_toks['input_ids'],
            #         attention_mask=new_prompt_toks['attention_mask'],
            #         max_length=new_prompt_toks['input_ids'].shape[1] + 20,
            #         pad_token_id=tokenizer_p.pad_token_id,
            #         do_sample=False
            #     )
            #     gen_txt = tokenizer_p.decode(seq[0], skip_special_tokens=True)
            #     # print(gen_txt)
            #     # Extract the answer part
            #     reference_answer = extract_answer(gen_txt)
            #     reference_answer = strip_string(reference_answer)
            #     new_response = response + " The answer is " + reference_answer
            #     new_responses.append(new_response)
            #     # new_responses.append(gen_txt)

        tmp_data = {
            "prompt": prompt,
            "question": question,
            "responses": new_response,
            "answer": answer,
        }
        gathered_data.append(tmp_data)

    print(f"I collected {len(gathered_data)} samples")

    # Ensure the output directory exists
    # os.makedirs(script_args.output_dir, exist_ok=True)

    # output_file = os.path.join(
    #     script_args.output_dir, f"{script_args.local_index}.json"
    # )
    # with open(output_file, "w", encoding="utf8") as f:
    #     for sample in gathered_data:
    #         json.dump(sample, f, ensure_ascii=False)
    #         f.write('\n')

    # print(f"Data has been written to {output_file}")

    with open(script_args.output_dir + str(script_args.local_index) + ".json", "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write('\n')

if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args)
