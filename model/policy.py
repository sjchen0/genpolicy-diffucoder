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

        input_size = config.policy.hidden_size + 1
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
        
    def forward(self, hidden_states, log_score, time):
        # hidden_states: (B, L, h), log_score: (B, L, V+1), time: (B,)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            x = torch.cat([hidden_states, time[:,None,None].repeat(1, hidden_states.shape[1], 1)], dim=-1)
            x = self.mlp(x)
            x = F.softmax(x, dim=1)
        return x