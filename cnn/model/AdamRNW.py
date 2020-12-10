import math
import torch
from torch.optim.optimizer import Optimizer, required

class AdamRNW(Optimizer):
    r"""Implements AdamRNW algorithm.

    Rewrite
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        
        #defaults = dict(lr=lr, betas=betas, eps=eps,
        #                weight_decay=weight_decay)
        super(AdamRNW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamRNW, self).__setstate__(state)
        #for group in self.param_groups:
        #    group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Perform optimization step
                #grad = p.grad
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                p_data_fp32 = p.data.float()
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0 # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32) # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                state['step'] += 1
                # Decay the first and second moment running average coefficient
                #bias_correction1 = 1 - beta1 ** state['step']
                #bias_correction2 = 1 - beta2 ** state['step']
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t0 = beta2 ** state['step']
                    beta2_t1 = beta2_t0 * beta2
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma_t0 = N_sma_max - 2 * state['step'] * beta2_t0 / (1 - beta2_t0)
                    N_sma_t1 = N_sma_max - 2 * (state['step']+1) * beta2_t1 / (1 - beta2_t1)
                    buffered[1] = N_sma_t0

                    # more conservative since it's an approximated value
                    if N_sma_t0 >= 5:
                        step_size_t0 = math.sqrt((1 - beta2_t0) * (N_sma_t0 - 4) / (N_sma_max - 4) * (N_sma_t0 - 2) / N_sma_t0 * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        step_size_t1 = math.sqrt((1 - beta2_t1) * (N_sma_t1 - 4) / (N_sma_max - 4) * (N_sma_t1 - 2) / N_sma_t1 * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** (state['step']+1))
                        
                    elif self.degenerated_to_sgd:
                        step_size_t0 = 1.0 / (1 - beta1 ** state['step'])
                        step_size_t1 = 1.0 / (1 - beta1 ** (state['step']+1))
                    else:
                        step_size_t0 = -1
                        step_size_t1 = -1
                    buffered[2] = step_size_t0

                # more conservative since it's an approximated value
                if N_sma_t0 >= 5:
                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    # Perform grad update
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size_t0 * group['lr'] * (1 - beta1), grad, denom)
                    p_data_fp32.addcdiv_(-step_size_t1 * group['lr'] * beta1 , exp_avg, denom)
                    # Copy back
                    p.data.copy_(p_data_fp32)
                elif step_size_t0 > 0:
                    # Perform stepweight decay
                    p_data_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                    # Perform grad update
                    p_data_fp32.add_(-step_size_t0 * group['lr'] * (1 - beta1), grad)
                    p_data_fp32.add_(-step_size_t1 * group['lr'] * beta1, exp_avg)
                    # Copy back
                    p.data.copy_(p_data_fp32)

        return loss    