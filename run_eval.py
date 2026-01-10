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

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
model_path = "apple/DiffuCoder-7B-Instruct"
policy_path = "../output/2026.01.08/031408/checkpoints-meta/checkpoint.pth"
policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), policy_path)

model, tokenizer = load_customized_model_and_tokenizer(model_path, dtype="bfloat16")
policy = load_policy(policy_path, device="cuda")
model = torch.compile(model)
policy = torch.compile(policy)
lm = DiffucoderLM(model, policy, tokenizer, device="cuda")

results = evaluator.simple_evaluate(
    model=lm,
    tasks=["humaneval"],
    num_fewshot=0,
    batch_size=1,
    confirm_run_unsafe_code=True,
)

pprint(results)
logging.info(pformat(results))