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
policy_path = "../output/2026.01.16/061716/checkpoints/checkpoint_100.pth"
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
    logging.info(pformat(results))