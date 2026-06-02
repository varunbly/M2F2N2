import torch
from copy import deepcopy
from .base import BaseMetaMethod
from .utils import task_loss

class ANIL(BaseMetaMethod):
    def train_step(self, model, support, query, inner_lr, inner_steps, lam):
        saved = deepcopy(model.state_dict())
        head_names = model.head_param_names()

        for _ in range(inner_steps):
            for b in support:
                loss, _, _ = task_loss(model, b, lam)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in head_names and p.grad is not None:
                            p.data -= inner_lr * p.grad

        q_loss_total, n = 0.0, 0
        for b in query:
            l, _, _ = task_loss(model, b, lam)
            q_loss_total += l
            n += 1
        q_loss = q_loss_total / max(n, 1)

        model.zero_grad()
        q_loss.backward()
        meta_grads = {n: p.grad.clone() if p.grad is not None else None
                      for n, p in model.named_parameters()}

        model.load_state_dict(saved)
        for n, p in model.named_parameters():
            p.grad = meta_grads.get(n)

        return q_loss.item()
        
    def adapt(self, model, support, inner_lr, inner_steps, lam):
        adapted = deepcopy(model)
        head_names = adapted.head_param_names()
        adapted.train()
        for _ in range(inner_steps):
            for b in support:
                loss, _, _ = task_loss(adapted, b, lam)
                adapted.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for n, p in adapted.named_parameters():
                        if p.grad is None:
                            continue
                        if n not in head_names:
                            continue
                        p.data -= inner_lr * p.grad
        return adapted
        
    def predict(self, adapted_state, price, news, tf):
        adapted_state.eval()
        with torch.no_grad():
            pos, _ = adapted_state(price, news, tf, compute_align=False)
        return pos
