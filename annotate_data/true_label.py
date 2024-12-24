import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator
import re
import string
from datasets import Dataset
tqdm.pandas()
from grader import math_equal
from parser import extract_answer, strip_string
import argparse

def parse_arguments():
    """
    Parse command-line arguments for a script.

    Returns:
        Namespace: Parsed arguments with attributes `ds_dir`, `output_dir`, and `repo_id`.
    """
    parser = argparse.ArgumentParser(description="Process directories and repository ID.")
    parser.add_argument(
        '--ds_dir', 
        type=str, 
        required=True, 
        help="Path to the dataset directory."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Path to the output directory."
    )
    parser.add_argument(
        '--repo_id', 
        type=str, 
        required=False, 
        help="ID of the repository."
    )
    return parser.parse_args()

# Token name: For repo Iterative-DPO(write)
os.environ["HF_TOKEN"] = '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'
args = parse_arguments()
# ds_dir = "/home/swb9572/iterative-dpo/data/basesft_iter3/merge_data.json"
ds_dir = args.ds_dir
ds = load_dataset("json", data_files=ds_dir, split="train")
print(ds)



def tokenize(sample):
    preprocessed_resps = [resp.lstrip() for resp in sample["responses"]]
    ex_answer = [strip_string(extract_answer(resp)) for resp in preprocessed_resps]
    result = [math_equal(ex_answer[i], sample["answer"], include_percentage=True, is_close=True, timeout=True) for i in range(len(preprocessed_resps))]
    # print(result)
    if (all(result)) or (not any(result)):
        sample['pick'] = False
        sample['rewards'] = [0.0]
    else:
        sample['pick'] = True
        new_rewards = [1.0 if y else -1.0 for y in result]
        sample['rewards'] = new_rewards
    return sample

ds = ds.map(tokenize, num_proc=16)
new_dataset = ds.filter(lambda x: x['pick'] == True)
print(new_dataset)

# new_dataset.to_json("data/basesft_iter3/filtered_dataset.json")
# os.makedirs(args.output_dir, exist_ok=True)
new_dataset.to_json(args.output_dir)

# repo_id = "Yuanxin-Liu/Iter3_generation"  
if args.repo_id:
    new_dataset.push_to_hub(args.repo_id)

