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


class SimpleTransformerBlock(nn.Module, PyTorchModelHubMixin):
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

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -float("inf"))

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                -float("inf")
            )

        attn = torch.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        x = x + self.attn(h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
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
            if "attn_mask" in kwargs:
                x = self.tf(x, attn_mask=kwargs["attn_mask"])
            else:
                x = self.tf(x)
            x = self.final_layer(x)
            suppress_mask = torch.cat([forward_suppress.unsqueeze(-1), backward_suppress.unsqueeze(-1)], dim=-1)
            # x = x.masked_fill(suppress_mask, -float("inf"))
            x = F.softmax(x, dim=1)
        return x