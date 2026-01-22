import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import default_data_collator
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model.utils import load_policy
from datasets import load_dataset, concatenate_datasets
from omegaconf import OmegaConf
from model.policy import PolicyNet, PolicyTransformer
import losses
import wandb
import os
from datetime import datetime
import utils
from data import tokenizer_fn
from pathlib import Path

def load_customized_model_and_tokenizer(cfg):
    model_path = cfg.pretrained_hf_path
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = DreamModel(model_config)
    hf_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    state_dict = hf_model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if cfg.policy.dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    return model, tokenizer

def load_model_and_tokenizer(cfg):
    model_path = cfg.pretrained_hf_path
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def run(cfg):
    work_dir = cfg.work_dir
    # cfg = OmegaConf.load("configs/config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")

    utils.makedirs(checkpoint_dir)
    utils.makedirs(os.path.dirname(checkpoint_meta_dir))
    wandb.init(dir=os.path.abspath(work_dir), project='diffucoder-policy', config=OmegaConf.to_container(cfg, resolve=True), name=cfg.wandb_name, job_type='train')
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = model.to(device).eval()
    logger.info(f"Loaded model dtype {model.dtype}, size {model.num_parameters()}")

    keep_columns = ["instruction", "output"]

    educational_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")["train"]
    evol_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct")["train"]
    mceval_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct")["train"]
    package_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct")["train"]

    train_dataset = concatenate_datasets([
        educational_instruct.remove_columns(list(set(educational_instruct.column_names) - set(keep_columns))),
        evol_instruct.remove_columns(list(set(evol_instruct.column_names) - set(keep_columns))),
        mceval_instruct.remove_columns(list(set(mceval_instruct.column_names) - set(keep_columns))),
        package_instruct.remove_columns(list(set(package_instruct.column_names) - set(keep_columns)))
    ])

    train_dataset = educational_instruct

    tokenized_dataset = train_dataset.map(
        tokenizer_fn, 
        remove_columns=train_dataset.column_names, 
        batched=False, 
        num_proc=8, 
        desc="Tokenizing dataset", 
        load_from_cache_file=False
    )
    train_loader = DataLoader(tokenized_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=default_data_collator)
    logger.info(f"Loaded dataset with {len(tokenized_dataset)} samples")

    # initialize new policy
    policy = PolicyTransformer(cfg).to(device)

    # load pre-trained policy
    # policy_path = "../output/2026.01.18/192854/checkpoints/checkpoint_700.pth"
    # policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), policy_path)
    # policy_weights = torch.load(Path(policy_path), weights_only=False, map_location=device)['policy_model']
    # policy.load_state_dict(policy_weights)

    logger.info(f"Policy model dtype {policy.dtype}, size {sum(p.numel() for p in policy.parameters())}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    discrete_timesteps = torch.linspace(0, 1, steps=cfg.sampling.discrete_steps + 2, device=device)[1:-1]
    discrete_timesteps = discrete_timesteps ** (cfg.sampling.discrete_time_exponent)

    optimize_fn = losses.optimization_manager(cfg)
    special_tokens = dict(mask_token_id = model.config.mask_token_id, pad_token_id=tokenizer.pad_token_id)
    train_step_fn = losses.get_policy_step_fn(None, special_tokens, True, discrete_timesteps, optimize_fn, cfg.training.accum, cfg.training.loss_type)
    eval_step_fn = losses.get_policy_step_fn(None, special_tokens, False, discrete_timesteps, optimize_fn, cfg.training.accum, cfg.training.loss_type)

    state = dict(optimizer=optimizer, score_model=model, policy_model=policy, scaler=scaler, step=0)

    for epoch in range(cfg.training.n_epoch):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if batch["prompt_mask"].sum(dim=-1).max() >= batch["input_ids"].shape[1] - cfg.sampling.discrete_steps - 1:
                continue
            loss = train_step_fn(state, batch)
            if state["step"] % cfg.training.log_freq == 0:
                logger.info(f"Epoch {epoch}, Step {state['step']}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()}, step = state['step'] + len(train_loader) * epoch)
            if state["step"] % cfg.training.snapshot_freq == 0:
                utils.save_policy_checkpoint(checkpoint_meta_dir, state)
                utils.save_policy_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{state["step"]}.pth'), state)

    wandb.finish()