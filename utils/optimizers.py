import torch
from torch.optim import Optimizer
from typing import Iterable, Callable, Optional


class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0.01, amsgrad: bool = False, maximize: bool = False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if maximize:
                    grad = -grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # w = w * (1 - lr * weight_decay)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                else:
                    # denom = sqrt(v_t / bias_correction2) + eps
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                # w = w - step_size * m_t / denom
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def get_state_dict_for_logging(self) -> dict:
        info = {
            "optimizer": "AdamW (custom)",
            "param_groups": []
        }
        for i, group in enumerate(self.param_groups):
            info["param_groups"].append({
                "group_id": i,
                "lr": group["lr"],
                "betas": group["betas"],
                "eps": group["eps"],
                "weight_decay": group["weight_decay"],
                "amsgrad": group["amsgrad"],
            })
        return info


class AdamWWithWarmup(AdamW):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = 1000,
        min_lr: float = 0.0,
        scheduler_type: str = "cosine",
    ):
        super().__init__(params, lr, betas, eps, weight_decay)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = lr
        self.scheduler_type = scheduler_type
        self._current_step = 0

    def get_lr_multiplier(self) -> float:
        if self._current_step < self.warmup_steps:
            return self._current_step / max(1, self.warmup_steps)
        else:
            progress = (self._current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)

            if self.scheduler_type == "cosine":
                import math
                return self.min_lr / self.base_lr + (1 - self.min_lr / self.base_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                return 1.0 - progress * (1 - self.min_lr / self.base_lr)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        lr_mult = self.get_lr_multiplier()
        for group in self.param_groups:
            group["lr"] = self.base_lr * lr_mult

        self._current_step += 1
        return super().step(closure)

