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


class TransformerBlock(nn.Module, PyTorchModelHubMixin):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-norm
        h = self.ln1(x)
        a, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,                 # e.g. causal mask [T,T] or [B*nheads,T,T] in some cases
            key_padding_mask=key_padding_mask,   # [B,T] True for PAD positions
            need_weights=False
        )
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class PolicyTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if type(config) == dict:
            config = OmegaConf.create(config)
        self.config = config
        self.input_size = config.policy.hidden_size + 2 # two additional masking embeddings
        self.n_heads = 4
        self.attn_hidden_size = self.input_size // self.n_heads * self.n_heads
        self.ffn_hidden_size = self.input_size * 2

        self.first_layer = nn.Linear(self.input_size, self.attn_hidden_size)

        self.tf = TransformerBlock(
            d_model=self.attn_hidden_size,
            n_heads=self.n_heads,
            d_ff=self.ffn_hidden_size
        )

        self.final_layer = nn.Linear(self.attn_hidden_size, 2)

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
            forward_suppress = kwargs["prompt_index"] + kwargs["mask_index"]
            backward_suppress = kwargs["prompt_index"] + ~kwargs["mask_index"]
            x = torch.cat([
                hidden_states,
                forward_suppress.unsqueeze(-1).to(hidden_states.dtype),
                backward_suppress.unsqueeze(-1).to(hidden_states.dtype),
            ], dim=-1)
            x = self.first_layer(x)
            x = self.tf(x, attn_mask=kwargs["attn_mask"])
            x = self.final_layer(x)
            suppress_mask = torch.cat([forward_suppress.unsqueeze(-1), backward_suppress.unsqueeze(-1)], dim=-1)
            x.masked_fill(suppress_mask, -float("inf"))
            x = F.softmax(x, dim=1)
        return x