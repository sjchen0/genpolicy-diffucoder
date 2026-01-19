import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

class PolicyNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config
        input_size = config.policy.hidden_size + 2
        mlp_hidden = config.policy.hidden_size // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2)
        )
        if config.policy.dtype == 'float32':
            self.dtype = torch.float32
        elif config.policy.dtype == 'float16':
            self.dtype = torch.float16
        elif config.policy.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
        
    def forward(self, hidden_states, log_score, time, **kwargs):
        '''
        hidden_states: (B, L, h), 
        log_score: (B, L, V+1), 
        time: (B,), 
        mask_index: (B, L), bool
        prompt_index: (B, L), bool
        '''
        with torch.cuda.amp.autocast(dtype=self.dtype):
            if "mask_index" in kwargs and "prompt_index" in kwargs: # ignore time
                forward_suppress = kwargs["prompt_index"] + kwargs["mask_index"]
                backward_suppress = kwargs["prompt_index"] + ~kwargs["mask_index"]
                x = torch.cat([
                    hidden_states,
                    forward_suppress.unsqueeze(-1).to(hidden_states.dtype),
                    backward_suppress.unsqueeze(-1).to(hidden_states.dtype),
                ], dim=-1)
                x = self.mlp(x)
                suppress_mask = torch.cat([forward_suppress.unsqueeze(-1), backward_suppress.unsqueeze(-1)], dim=-1)
                x.masked_fill(suppress_mask, -float("inf"))
                x = F.softmax(x, dim=1)
            else: # use time
                x = torch.cat([hidden_states, time[:,None,None].repeat(1, hidden_states.shape[1], 1)], dim=-1)
                x = self.mlp(x)
                x = F.softmax(x, dim=1)
        return x