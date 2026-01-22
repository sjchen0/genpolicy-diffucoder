import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def Batch_Uniform_Sampler(B, type = 'naive', device = 'cuda'):
    def vdm_sampler(B, device):
        u_0 = torch.rand(1, device=device)  # Sample u_0 from U(0, 1)
        t = [(u_0 + i / B) % 1 for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    
    def decoupled_sampler(B, device):
        u = torch.rand(B, device=device)  # Sample B independent values from U(0, 1)
        t = [(u[i] + i) / B for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    if type == 'naive':
        return torch.rand(B, device = device)
    elif type == 'vdm':
        return vdm_sampler(B, device)
    elif type == 'decoupled':
        return decoupled_sampler(B, device)
    else:
        raise ValueError(f"{type} not valid")
    

def get_policy_loss_fn(noise, special_tokens, train, discrete_timesteps, num_trajectories=2, sampling_eps=1e-3, loss_type='lambda_DCE', order = torch.arange(1024)):

    def policy_log_loss_bidirection(score_model, policy_model, batch, cond = None):
        # 1. given a batch, sample multiple trajectories at discrete timesteps 0 < t1 < ... < tK < 1
        # 2. at each state of each trajectory, looking at the unmasked places,
        #    evaluate (log_policy_model(x_k) + log_score_model(x_k-1 | x_k)).sum().exp()
        # 3. then aggregate and take negative log
        
        mask_token_id = special_tokens['mask_token_id']
        pad_token_id = special_tokens['pad_token_id']

        input_ids, attention_mask, prompt_mask = batch["input_ids"], batch["attention_mask"], batch["prompt_mask"]

        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = attention_mask.to(torch.bfloat16)

        # EXPERIMENTAL: attending to PAD
        # attention_mask = None

        score_model.eval()
        if train:
            policy_model.train()
        else:
            policy_model.eval()

        total_loss = torch.zeros(input_ids.shape[0], device=input_ids.device)
        total_loss_raw = torch.zeros(input_ids.shape[0], device=input_ids.device)
        B, L = input_ids.shape
        response_len = L - prompt_mask.sum(dim=-1)
        min_response_len = L - prompt_mask.sum(dim=-1).max()
        K = discrete_timesteps.shape[0]
        
        nums_unmask = response_len // K
        num_unmask = min_response_len // K

        for n in range(num_trajectories):
            # sample discrete timesteps
            input_ids_km1 = input_ids.clone()
            forward_policy = None
            x0_hidden_state = None
            for k in range(K):
                with torch.no_grad():
                    out = score_model(input_ids_km1, attention_mask, output_hidden_states=True, return_dict=True)
                    log_condition, hidden_state = out.logits, out.hidden_states[-1]
                    # EXPERIMENTAL: shift hidden_state by 1
                    # hidden_state = torch.concat([hidden_state[:, :1], hidden_state[:, :-1]], dim=1)
                    x0_hidden_state = hidden_state

                log_condition = torch.cat([log_condition[:,:1], log_condition[:, :-1]], dim=1)
                input_ids_t = discrete_timesteps[k] * torch.ones(B, device=input_ids.device)
                mask = input_ids_km1 == mask_token_id
                mask = torch.logical_or(mask, prompt_mask == 1)
                forward_policy = policy_model(
                    x0_hidden_state,
                    log_condition,
                    input_ids_t,
                    mask_index=(input_ids_km1 == mask_token_id),
                    prompt_index=(prompt_mask == 1),
                    attn_mask=attention_mask.squeeze(1)
                )[:,:,0] # (B, L)

                forward_set = torch.zeros_like(input_ids, dtype=torch.bool)
                forward_indices_list = [
                    torch.zeros((nums_unmask[b]), dtype=torch.int64, device=forward_policy.device) 
                    for b in range(forward_policy.shape[0])
                ]
                for b in range(forward_policy.shape[0]):
                    idx = torch.multinomial(forward_policy[b], nums_unmask[b], replacement=False)
                    # if forward_policy[b].sum() > 0:
                    #     idx = torch.multinomial(forward_policy[b], nums_unmask[b], replacement=False)
                    # else:
                    #     idx = torch.multinomial(torch.ones_like(forward_policy[b]), nums_unmask[b], replacement=False)
                    forward_indices_list[b] = idx
                    forward_set[b].scatter_(0, forward_indices_list[b], True)

                log_forward_prob = ((forward_policy + 1e-20).log() * forward_set).mean(-1)

                input_ids_k = input_ids_km1.clone()
                input_ids_k[forward_set] = mask_token_id

                with torch.no_grad():
                    out = score_model(input_ids_k, attention_mask, output_hidden_states=True, return_dict=True)
                    log_condition, hidden_state = out.logits, out.hidden_states[-1]
                    # EXPERIMENTAL: shift hidden_state by 1
                    # hidden_state = torch.concat([hidden_state[:, :1], hidden_state[:, :-1]], dim=1)

                # reset vocab_probs according to Dream config
                log_condition = torch.cat([log_condition[:,:1], log_condition[:, :-1]], dim=1)
                vocab_probs = F.softmax(log_condition, dim=-1)

                unmasked = (input_ids_k != input_ids_km1).to(vocab_probs.dtype)
                # EXPERIMENTAL: ignore PADs
                # unmasked = unmasked * (input_ids_k != pad_token_id).to(unmasked.dtype)
                target_onehot = F.one_hot(input_ids_km1, num_classes=vocab_probs.shape[-1])
                vocab_probs = (vocab_probs * target_onehot).sum(-1) # (B, L)
                mask = (input_ids_k == mask_token_id)
                mask = torch.logical_or(mask, prompt_mask == 1)
                policy_out = policy_model(
                    hidden_state,
                    log_condition,
                    input_ids_t,
                    mask_index=(input_ids_k == mask_token_id),
                    prompt_index=(prompt_mask == 1),
                    attn_mask=attention_mask.squeeze(1)
                )
                
                backward_policy = policy_out[:,:,1]
                log_step_metric = ((
                    (vocab_probs + 1e-20).log() + (backward_policy + 1e-20).log() #- (0.01 * (forward_policy + 1e-20)).log()
                ) * unmasked).mean(-1)
                # print(log_step_metric, vocab_probs.min(), backward_policy.min(), forward_policy.min())
                
                # forward_policy = policy_out[:,:,0]
                
                # REINFORCE unbiased gradient estimate
                total_loss -= (log_step_metric + (log_step_metric.detach() - log_step_metric.detach().mean()) * log_forward_prob)
                total_loss_raw -= log_step_metric

                input_ids_km1 = input_ids_k.clone()

                # import ipdb; ipdb.set_trace()

        return (total_loss, total_loss_raw)
    
    if loss_type == 'policy_log_loss':
        return policy_log_loss_bidirection
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported yet!')


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_policy_step_fn(noise, special_tokens, train, discrete_timesteps, optimize_fn, accum, loss_type):
    policy_loss_fn = get_policy_loss_fn(noise, special_tokens, train, discrete_timesteps, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0
    total_loss_raw = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss
        nonlocal total_loss_raw

        score_model = state['score_model']
        policy_model = state['policy_model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss_tuple = policy_loss_fn(score_model, policy_model, batch, cond=cond)
            
            loss, loss_raw = loss_tuple[0].mean() / accum, loss_tuple[1].mean() / accum

            # print(loss)
            scaler.scale(loss).backward()

            # check gradient
            # for name, param in policy_model.named_parameters():
            #     print(name, param.requires_grad)
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean().item())
            #     else:
            #         print(name, "has no grad")


            accum_iter += 1
            total_loss += loss.detach()
            total_loss_raw += loss_raw.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, policy_model.parameters(), step=state['step'])
                # state['ema'].update(policy_model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                loss_raw = total_loss_raw

                total_loss_raw = 0
                total_loss = 0
        else:
            with torch.no_grad():
                # ema = state['ema']
                # ema.store(policy_model.parameters())
                # ema.copy_to(policy_model.parameters())
                loss_tuple = policy_loss_fn(score_model, policy_model, batch, cond=cond)
                loss = loss_tuple[0].mean()
                loss_raw = loss_tuple[1].mean()
                # ema.restore(policy_model.parameters())

        return loss_raw

    return step_fn