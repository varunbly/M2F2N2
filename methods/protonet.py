import torch
import torch.nn.functional as F
from .base import BaseMetaMethod
from .utils import task_loss

class ProtoNet(BaseMetaMethod):
    def train_step(self, model, support, query, inner_lr, inner_steps, lam):
        model.train()
        support_embs, support_pos = [], []
        for b in support:
            price, news, tf, tar = b
            hp, hn, ht = model._encode(price, news, tf)
            h_fused = model.fusion(hp, hn, ht)
            ctx = model._temporal_aggregate(h_fused)
            p_ctx = model._temporal_aggregate(hp)
            emb = torch.cat([ctx, p_ctx], dim=-1)
            support_embs.append(emb)
            support_pos.append(tar)
            
        if not support_embs:
            q_loss_total, n = 0.0, 0
            model.zero_grad()
            for b in query:
                loss, _, _ = task_loss(model, b, lam)
                loss.backward()
                q_loss_total += loss.item()
                n += 1
            return q_loss_total / max(n, 1)

        S_emb = torch.cat(support_embs, dim=0)
        S_pos = torch.cat(support_pos, dim=0)
        
        q_loss_total, n = 0.0, 0
        model.zero_grad()
        
        total_loss = 0.0
        for b in query:
            price, news, tf, target = b
            hp, hn, ht = model._encode(price, news, tf)
            al = model._infonce(hp, hn)
            
            h_fused = model.fusion(hp, hn, ht)
            ctx = model._temporal_aggregate(h_fused)
            p_ctx = model._temporal_aggregate(hp)
            Q_emb = torch.cat([ctx, p_ctx], dim=-1)
            
            dists = torch.cdist(Q_emb, S_emb)
            weights = F.softmax(-dists, dim=1)
            
            pos = weights @ S_pos
            
            pred_loss = F.mse_loss(pos, target)
            loss = pred_loss + lam * al
            
            total_loss = total_loss + loss
            q_loss_total += loss.item()
            n += 1

        if n > 0:
            total_loss = total_loss / n
            total_loss.backward()

        return q_loss_total / max(n, 1)

    def adapt(self, model, support, inner_lr, inner_steps, lam):
        model.eval()
        S_emb_list, S_pos_list = [], []
        with torch.no_grad():
            for b in support:
                price, news, tf, tar = b
                hp, hn, ht = model._encode(price, news, tf)
                h_fused = model.fusion(hp, hn, ht)
                ctx = model._temporal_aggregate(h_fused)
                p_ctx = model._temporal_aggregate(hp)
                emb = torch.cat([ctx, p_ctx], dim=-1)
                S_emb_list.append(emb)
                S_pos_list.append(tar)
        if S_emb_list:
            S_emb = torch.cat(S_emb_list, dim=0)
            S_pos = torch.cat(S_pos_list, dim=0)
        else:
            S_emb, S_pos = None, None
        return (model, S_emb, S_pos)

    def predict(self, adapted_state, price, news, tf):
        model, S_emb, S_pos = adapted_state
        if S_emb is None:
            pos, _ = model(price, news, tf, compute_align=False)
            return pos
            
        model.eval()
        with torch.no_grad():
            hp, hn, ht = model._encode(price, news, tf)
            h_fused = model.fusion(hp, hn, ht)
            ctx = model._temporal_aggregate(h_fused)
            p_ctx = model._temporal_aggregate(hp)
            Q_emb = torch.cat([ctx, p_ctx], dim=-1)
            
            dists = torch.cdist(Q_emb, S_emb)
            weights = F.softmax(-dists, dim=1)
            pos = weights @ S_pos
        return pos
