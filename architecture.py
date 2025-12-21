import os, math
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


class SpanHead(nn.Module):
    """
    Shared span representation -> scalar span score + type logits.
    """
    def __init__(self, span_hidden_size: int, num_types: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(span_hidden_size, span_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.span_score = nn.Linear(span_hidden_size, 1)     # scalar score
        self.type_head  = nn.Linear(span_hidden_size, num_types)  # type logits

    def forward(self, span_repr: torch.Tensor):
        h = self.ff(span_repr)
        scores = self.span_score(h).squeeze(-1)   # (N,)
        type_logits = self.type_head(h)           # (N, K)
        return scores, type_logits

class PartitionSpanSegmenter(nn.Module):
    """
    Encoder + span scorer trained as:
      For each gold segment k:
        - fix start token s_k (gold)
        - score all candidate ends e in [s_k .. s_k + max_span_len - 1]
        - CE loss to pick gold end e_k among those candidates
      Plus type loss on the gold span itself.
    """
    def __init__(self, model_name: str, num_types: int = 3, max_span_len: int = 50):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.max_span_len = max_length = max_span_len
        self.span_head = SpanHead(span_hidden_size=hidden * 4, num_types=num_types)

    def _span_repr(self, h_b, cls_b, s: int, e: int):
        """
        Build span representation: [h_start, h_end, mean(h_s..h_e), cls]
        """
        h_start = h_b[s]
        h_end   = h_b[e]
        h_mean  = h_b[s:e+1].mean(dim=0)
        return torch.cat([h_start, h_end, h_mean, cls_b], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts=None,   # list[list[int]] gold starts
        span_ends=None,     # list[list[int]] gold ends
        span_types=None,    # list[list[int]] gold types
        type_loss_weight: float = 1.0,
    ):
        """
        Returns a dict with:
          loss, loss_end, loss_type
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state              # (B, T, H)
        cls = h[:, 0]                          # (B, H)

        device = h.device
        ce = nn.CrossEntropyLoss()

        # Initialize with a 'graph anchor' to ensure the loss remains 
        # differentiable even if no segments are found in a batch.
        graph_anchor = h.sum() * 0.0
        total_end_loss = graph_anchor
        total_type_loss = graph_anchor
        n_seg = 0

        B, T, _ = h.shape

        # If no gold spans provided (inference mode)
        if span_starts is None or span_ends is None or span_types is None:
            return {"loss": None}

        for b in range(B):
            h_b = h[b]
            cls_b = cls[b]

            valid_len = int(attention_mask[b].sum().item())
            last_tok = max(1, valid_len - 2)

            gs_list = list(map(int, span_starts[b]))
            ge_list = list(map(int, span_ends[b]))
            gt_list = list(map(int, span_types[b]))

            if len(gs_list) == 0:
                continue

            gold = sorted(zip(gs_list, ge_list, gt_list), key=lambda x: x[0])

            for (s_gold, e_gold, t_gold) in gold:
                # Basic boundary validation
                if s_gold < 1 or s_gold > last_tok:
                    continue
                if e_gold < s_gold or e_gold > last_tok:
                    continue

                # Candidate ends logic
                e_max = min(s_gold + self.max_span_len - 1, last_tok)
                cand_ends = list(range(s_gold, e_max + 1))

                gold_idx = e_gold - s_gold 
                if gold_idx < 0 or gold_idx >= len(cand_ends):
                    continue

                # Build reprs and get scores
                span_reprs = []
                for e in cand_ends:
                    span_reprs.append(self._span_repr(h_b, cls_b, s_gold, e))
                span_reprs = torch.stack(span_reprs, dim=0)

                cand_scores, cand_type_logits = self.span_head(span_reprs)

                # End selection loss
                end_loss = ce(cand_scores.unsqueeze(0), torch.tensor([gold_idx], device=device))

                # Type loss (gold span only)
                type_logits_gold = cand_type_logits[gold_idx].unsqueeze(0)
                type_loss = ce(type_logits_gold, torch.tensor([t_gold], device=device))

                total_end_loss += end_loss
                total_type_loss += type_loss
                n_seg += 1

        # Final loss reduction
        if n_seg > 0:
            loss_end = total_end_loss / n_seg
            loss_type = total_type_loss / n_seg
            loss = loss_end + type_loss_weight * loss_type
        else:
            loss_end = graph_anchor
            loss_type = graph_anchor
            loss = graph_anchor
            
            if self.training:
                print(f"Warning: Batch with 0 segments encountered. Loss is zeroed but graph is maintained.")

        return {
            "loss": loss,
            "loss_end": loss_end.detach(),
            "loss_type": loss_type.detach(),
        }
        
    def predict_span(self, h_b, cls_b, s, e):
        """
        Calculates the score and type for a specific candidate span (s, e).
        """
        repr_ = self._span_repr(h_b, cls_b, s, e)
        # span_head returns (scores, type_logits)
        score, type_logits = self.span_head(repr_.unsqueeze(0))
        pred_type = torch.argmax(type_logits, dim=-1).item()
        return score.item(), pred_type