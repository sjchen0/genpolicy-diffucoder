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
    

def get_policy_loss_fn(noise, token_dim, train, discrete_timesteps, num_trajectories=1, sampling_eps=1e-3, loss_type='lambda_DCE', order = torch.arange(1024)):

    def policy_log_loss_bidirection(score_model, policy_model, batch, cond = None):
        # 1. given a batch, sample multiple trajectories at discrete timesteps 0 < t1 < ... < tK < 1
        # 2. at each state of each trajectory, looking at the unmasked places,
        #    evaluate (log_policy_model(x_k) + log_score_model(x_k-1 | x_k)).sum().exp()
        # 3. then aggregate and take negative log
        
        input_ids, attention_mask, prompt_mask = batch["input_ids"], batch["attention_mask"], batch["prompt_mask"]

        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = attention_mask.to(torch.bfloat16)

        score_model.eval()
        if train:
            policy_model.train()
        else:
            policy_model.eval()

        total_loss = torch.zeros(input_ids.shape[0], device=input_ids.device)
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
            for k in range(K):
                with torch.no_grad():
                    out = score_model(input_ids_km1, attention_mask, output_hidden_states=True, return_dict=True)
                    log_condition, hidden_state = out.logits, out.hidden_states[-1]

                input_ids_t = discrete_timesteps[k] * torch.ones(B, device=input_ids.device)
                if forward_policy is None:
                    forward_policy = policy_model(hidden_state, log_condition, input_ids_t)[:,:,0] # (B, L)
                mask = input_ids_km1 == token_dim - 1
                # TODO: take logical_or with prompt mask
                mask = torch.logical_or(mask, prompt_mask == 1)

                # forward_policy = forward_policy.masked_fill(mask, -1e9)
                # forward_policy = F.softmax(forward_policy, dim=-1)
                forward_policy = forward_policy.masked_fill(mask, 0.)

                forward_set = torch.zeros_like(input_ids, dtype=torch.bool)
                forward_indices_list = [
                    torch.zeros((nums_unmask[b]), dtype=torch.int64, device=forward_policy.device) 
                    for b in range(forward_policy.shape[0])
                ]
                # forward_indices = torch.zeros((forward_policy.shape[0], num_unmask), dtype=torch.int64, device=forward_policy.device)
                for b in range(forward_policy.shape[0]):
                    idx = torch.multinomial(forward_policy[b], nums_unmask[b], replacement=False)
                    # print("sampled indices len:", len(idx), idx)
                    forward_indices_list[b] = idx
                    forward_set[b].scatter_(0, forward_indices_list[b], True)

                log_forward_prob = ((forward_policy + 1e-20).log() * forward_set).mean(-1)
                # import ipdb; ipdb.set_trace()

                input_ids_k = input_ids_km1.clone()
                input_ids_k[forward_set] = token_dim - 1

                with torch.no_grad():
                    out = score_model(input_ids_k, attention_mask, output_hidden_states=True, return_dict=True)
                    log_condition, hidden_state = out.logits, out.hidden_states[-1]
                
                # vocab_probs = torch.ones_like(log_condition, dtype=policy_model.module.dtype)
                # vocab_probs[:,:,:-1] = F.softmax(log_condition[:,:,:-1], dim=-1) # (B, L, V)

                soft = F.softmax(log_condition[:,:,:-1], dim=-1)
                vocab_probs = torch.cat([soft, torch.ones_like(soft[..., :1])], dim=-1)

                unmasked = (input_ids_k != input_ids_km1).to(vocab_probs.dtype)
                target_onehot = F.one_hot(input_ids_km1, num_classes=vocab_probs.shape[-1])
                vocab_probs = (vocab_probs * target_onehot).sum(-1) # (B, L)
                policy_out = policy_model(hidden_state, log_condition, input_ids_t)
                forward_policy = policy_out[:,:,0]
                backward_policy = policy_out[:,:,1]

                mask = (input_ids_k == token_dim - 1)
                mask = torch.logical_or(mask, prompt_mask == 1)
                # backward_policy = backward_policy.masked_fill(mask, -1e9)
                
                # backward_policy = F.softmax(backward_policy, dim=-1)
                # backward_policy = backward_policy.masked_fill(mask, 0.)

                log_step_metric = (((vocab_probs + 1e-20).log() + (backward_policy + 1e-20).log()) * unmasked).mean(-1)
                
                # REINFORCE unbiased gradient estimate
                total_loss -= (log_step_metric + (log_step_metric.detach() - log_step_metric.detach().mean()) * log_forward_prob)

                input_ids_km1 = input_ids_k.clone()

                # import ipdb; ipdb.set_trace()

        return total_loss
    
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


def get_policy_step_fn(noise, token_dim, train, discrete_timesteps, optimize_fn, accum, loss_type):
    policy_loss_fn = get_policy_loss_fn(noise, token_dim, train, discrete_timesteps, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        score_model = state['score_model']
        policy_model = state['policy_model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = policy_loss_fn(score_model, policy_model, batch, cond=cond).mean() / accum
            
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
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, policy_model.parameters(), step=state['step'])
                # state['ema'].update(policy_model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                # ema = state['ema']
                # ema.store(policy_model.parameters())
                # ema.copy_to(policy_model.parameters())
                loss = policy_loss_fn(score_model, policy_model, batch, cond=cond).mean()
                # ema.restore(policy_model.parameters())

        return loss

    return step_fn