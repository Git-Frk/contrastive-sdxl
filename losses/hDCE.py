from packaging import version
import torch
from torch import nn


class PatchHDCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # positive logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        if self.args.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.args.train_batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # weighted by semantic relation
        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg - l_pos) / self.args.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v - v.detach())

        # for monitoring
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.args.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy,
                                               torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean() - 1 + CELoss_dummy.detach()

        return loss


def calculate_HDCE_loss(src, tgt, hDCE, args, nce_layers, weight=None):
    n_layers = len(nce_layers)
    crit = hDCE

    feat_q_pool = tgt
    feat_k_pool = src

    total_HDCE_loss = 0.0
    for f_q, f_k, w in zip(feat_q_pool, feat_k_pool, weight):
        if args.no_Hneg:
            w = None
        loss = crit(f_q, f_k, w) * args.lambda_HDCE
        total_HDCE_loss += loss.mean()

    return total_HDCE_loss / n_layers




