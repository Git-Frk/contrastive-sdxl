import torch
from torch import nn

from ..models.patch_network import Normalize
from ..schedules import LinearSchedule


SRC_sched = LinearSchedule()


class SRC_Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.normalize = Normalize()
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        self.kld = nn.KLDivLoss(reduction="sum", log_target=True)

    def forward(self, feat_q, feat_k, batch_size ,only_weight=False, epoch=None):
        '''
            param feat_q: target
            param feat_k: source
            return: SRC loss, weights for hDCE
        '''

        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = batch_size  
        feat_k = self.normalize(feat_k)
        feat_q = self.normalize(feat_q)

        # SRC
        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q_v.shape[1]
        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))

        weight_seed = spatial_k.clone().detach()
        diagonal = torch.eye(npatches, device=feat_k_v.device, dtype=torch.bool)[None, :, :]

        HDCE_gamma = self.args.HDCE_gamma
        if self.args.use_curriculum:
            HDCE_gamma = HDCE_gamma + (self.args.HDCE_gamma_min - HDCE_gamma) * (epoch) / (
                    self.args.max_train_epochs) # old: self.args.n_epochs + self.args.n_epochs_decay)
            if (self.args.step_gamma) and (epoch > self.args.step_gamma_epoch):
                HDCE_gamma = 1

        # weights by semantic relation
        weight_seed.masked_fill_(diagonal, -10.0)
        weight_out = self.softmax_dim2(weight_seed.clone() / HDCE_gamma).detach()
        wmax_out, _ = torch.max(weight_out, dim=2, keepdim=True)
        weight_out /= wmax_out

        if only_weight:
            return feat_q.new_zeros(()), weight_out

        spatial_q = self.softmax_dim1(spatial_q)
        spatial_k = self.softmax_dim1(spatial_k).detach()

        loss_src = self.get_jsd(spatial_q, spatial_k)/batch_size

        return loss_src, weight_out

    def get_jsd(self, p1, p2):
        '''
        param p1: n X C
        param p2: n X C
        return: n X 1
        '''
        m = 0.5 * (p1 + p2)
        return 0.5 * (self.kld(m.log(), p1.log()) + self.kld(m.log(), p2.log()))


def calculate_SRC_loss(src, tgt, nce_layers, src_loss, args, step, batch_size, only_weight=False, epoch=None):
    if len(src) != len(nce_layers) or len(tgt) != len(src):
        raise ValueError("Feature count does not match nce_layers")    
    
    n_layers = len(nce_layers)

    total_SRC_loss = 0.0
    weights = []
    for f_q, f_k in zip(tgt, src):
        loss_SRC, weight = src_loss(f_q, f_k, batch_size=batch_size, only_weight=only_weight, epoch=epoch)
        lambda_SRC = args.lambda_SRC * SRC_sched(step)
        total_SRC_loss += loss_SRC * lambda_SRC
        weights.append(weight)
    return total_SRC_loss / n_layers, weights
