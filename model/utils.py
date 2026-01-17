import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .modeling_dream import DreamModel
from .policy import PolicyNet
from hydra import initialize, initialize_config_dir, compose
from pathlib import Path

def load_customized_model_and_tokenizer(model_path, dtype="bfloat16"):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = DreamModel(model_config)
    hf_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    state_dict = hf_model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if dtype == "bfloat16":
        model = model.to(torch.bfloat16)
    del hf_model
    return model, tokenizer

def load_policy(policy_ckpt, device, use_init=False):
    if not use_init:
        policy_weights = torch.load(Path(policy_ckpt), weights_only=False, map_location=device)['policy_model']
        config_path = Path(policy_ckpt).parent.parent/".hydra"
        initialize_config_dir(config_dir=str(config_path), version_base=None)
        cfg = compose(config_name="config.yaml")
        policy_model = PolicyNet(cfg).to(device)
        policy_model.load_state_dict(policy_weights)
    else:
        config_path = Path(policy_ckpt).parent.parent/".hydra"
        initialize_config_dir(config_dir=str(config_path), version_base=None)
        cfg = compose(config_name="config.yaml")
        policy_model = PolicyNet(cfg).to(device)
    return policy_model