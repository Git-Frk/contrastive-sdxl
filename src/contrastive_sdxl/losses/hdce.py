import torch
from torch import nn

from ..schedules import LinearSchedule


hDCE_sched = LinearSchedule()


class HDCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q, feat_k, batch_size, weight=None):
        
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # positive logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm = batch_size
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # weighted by semantic relation
        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg - l_pos) / self.args.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v - v.detach())

        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.args.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy,
                                               torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean() - 1 + CELoss_dummy.detach()

        return loss


def calculate_HDCE_loss(src, tgt, hdce_loss, args, nce_layers, step, batch_size, weight=None):
    if len(src) != len(nce_layers) or len(tgt) != len(src):
        raise ValueError("Feature count does not match nce_layers")

    if weight is None:
        weight = [None] * len(tgt)

    if len(weight) != len(src):
        raise ValueError("Weight count does not match features")

    n_layers = len(nce_layers)

    total_HDCE_loss = 0.0

    for f_q, f_k, w in zip(tgt, src, weight):
        if args.no_Hneg:
            w = None
        lambda_HDCE = args.lambda_HDCE * hDCE_sched(step)
        loss = hdce_loss(f_q, f_k, batch_size=batch_size, weight=w) * lambda_HDCE
        total_HDCE_loss += loss.mean()

    return total_HDCE_loss / n_layers

