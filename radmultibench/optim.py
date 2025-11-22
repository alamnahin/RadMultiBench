import math
import torch
import torch.optim as optim


class WarmupLinearSchedule:
    """
    Implements the Noam LR scheduler (inverse Sqrt decay with linear warmup). 
    """

    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        t = self.current_step
        T_w = self.warmup_steps

    
        linear_warmup = min(1.0, t / T_w)
        inv_sqrt_decay = 1.0 / math.sqrt(max(t, T_w))
        lr = self.max_lr * linear_warmup * inv_sqrt_decay

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def build_optimizer(cfg, model):
    """Builds the optimizer."""
    params = filter(lambda p: p.requires_grad, model.parameters())

    if cfg.SOLVER.OPTIMIZER == "AdamW":
        return optim.AdamW(
            params,
            lr=cfg.SOLVER.LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.SOLVER.OPTIMIZER}")


def build_scheduler(cfg, optimizer):
    """Builds the learning rate scheduler."""
    return WarmupLinearSchedule(
        optimizer, warmup_steps=cfg.SOLVER.WARMUP_STEPS, max_lr=cfg.SOLVER.LR
    )
