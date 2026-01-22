import torch
from lm_eval import evaluator
from lm_eval.models.diffucoder import DiffucoderLM
from model.utils import load_customized_model_and_tokenizer, load_policy
import os
from pprint import pprint, pformat
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",
)

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = f"cuda:{local_rank}"

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
model_path = "apple/DiffuCoder-7B-Instruct"
# policy_path = "../output/2026.01.16/061716/checkpoints/checkpoint_100.pth" >>> 0.293 @ 128
# policy_path = "../output/2026.01.16/172450/checkpoints/checkpoint_200.pth" >>> 0.311 @ 128 
# policy_path = "../output/2026.01.18/192854/checkpoints/checkpoint_200.pth" >>> 0.293 @ 128
# policy_path = "../output/2026.01.18/192854/checkpoints/checkpoint_700.pth" >>> 0.16 @ 32
# policy_path = "../output/2026.01.19/184323/checkpoints/checkpoint_100.pth"

policy_path = "../output/2026.01.18/025924/checkpoints/checkpoint_700.pth" # trained with lambda=1
policy_path = "../output/2026.01.18/192854/checkpoints/checkpoint_700.pth" # trained with lambda=0
policy_path = "../output/2026.01.20/150143/checkpoints/checkpoint_200.pth" # continued with lambda=0.01

policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), policy_path)

model, tokenizer = load_customized_model_and_tokenizer(model_path, dtype="bfloat16")
policy = load_policy(policy_path, device=device)
model.policy_model = policy
# model = torch.compile(model)
# policy = torch.compile(policy)
lm = DiffucoderLM(model, policy, tokenizer, device=device)

results = evaluator.simple_evaluate(
    model=lm,
    tasks=["humaneval"],
    num_fewshot=0,
    batch_size=1,
    confirm_run_unsafe_code=True,
)

if int(os.environ.get("RANK", "0")) == 0:
    logging.info(pformat(results["results"]))
    pprint(results["results"])
    #logging.info(pformat(results))