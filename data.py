from transformers import AutoModel, AutoTokenizer, AutoConfig
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")
model_path = cfg.pretrained_hf_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def apply_chat_template(query, output):
    prompt = f"""<|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {query.strip()}
    <|im_end|>
    <|im_start|>assistant
    {output.strip()}
    <|im_end|>
    """
    return prompt

def format_example(example):
    query = example["instruction"]
    output = example["output"]
    return apply_chat_template(query, output)

def tokenizer_fn_no_prompt_mask(example):
    prompt = format_example(example)
    input = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=cfg.training.max_len,
        return_attention_mask=True
    )
    return {
        "input_ids": input.input_ids,
        "attention_mask": input.attention_mask
    }

def tokenizer_fn(example):
    prompt = format_example(example)
    input = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=cfg.training.max_len,
        return_attention_mask=True
    )
    
    prompt_end = prompt.find("<|im_start|>assistant")
    prompt_tokens = tokenizer(prompt[:prompt_end], truncation=True, max_length=cfg.training.max_len)
    prompt_mask = [1] * len(prompt_tokens.input_ids) + [0] * (cfg.training.max_len - len(prompt_tokens.input_ids))
    
    return {
        "input_ids": input.input_ids,
        "attention_mask": input.attention_mask,
        "prompt_mask": prompt_mask
    }