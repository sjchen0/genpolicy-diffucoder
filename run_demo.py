import torch
from model.utils import load_customized_model_and_tokenizer, load_policy
import os
from pprint import pprint, pformat
import logging

ckpt_num = 1000
use_init = False

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",
)

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = f"cuda:{local_rank}"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
model_path = "apple/DiffuCoder-7B-Instruct"
policy_path = "../output/2026.01.18/192854/checkpoints/checkpoint_700.pth"
policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), policy_path)

model, tokenizer = load_customized_model_and_tokenizer(model_path, dtype="bfloat16")
model = model.to(device=device).eval()
policy = load_policy(policy_path, device=device, use_init=use_init)

print(model.config.pad_token_id, tokenizer.pad_token_id, model.config.mask_token_id, tokenizer.mask_token_id)

query = """from typing import List


def has_close_elements(numbers: List[float], threshold: float)-> bool:
    \"\"\" 
    Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""
    
prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please complete the following problem:
```
{query}
```
<|im_end|>
<|im_start|>assistant
Here is the code to solve this problem:
```python
{query}
"""

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(device=device)
attention_mask = inputs.attention_mask.to(device=device)

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=True,
    steps=128,
    temperature=0.3,
    top_p=0.95,
    alg="policy",
    alg_temp=0.,
    policy_model=policy
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0])
print(generations[0].split('<|dlm_pad|>')[0])