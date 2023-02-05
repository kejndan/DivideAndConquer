import torch
import numpy as np

from Utils.log_utils import log_metrics


class MarginLoss(torch.nn.Module):
    def __init__(self, cfg, beta=1.2, margin=0.2, nu=0.0, dist_weighted_sampler_thr=0.5, device=None):
        super(MarginLoss, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor([beta]), requires_grad=True)
        self.margin = margin
        self.nu = nu
        self.thr = dist_weighted_sampler_thr
        self.device = cfg.device
        self.cfg = cfg

    @staticmethod
    def get_distance(x):
        mm = torch.mm(x, x.t())
        dist = mm.diag().view((mm.diag().size()[0], 1))
        dist = dist.expand_as(mm)
        dist_ = dist + dist.t()
        dist_ = (dist_ - 2 * mm).clamp(min=0)
        return dist_.clamp(min=1e-4).sqrt()

    def sample_triplets(self, embeddings, prod_labels, iter_=None, learner_id=None, for_cluster=False):
        anchor_ids, pos_ids, neg_ids = [], [], []
        if not torch.is_tensor(prod_labels):
            prod_labels = torch.tensor(prod_labels)

        distance = self.get_distance(embeddings)
        p0 = prod_labels.clone().view(1, prod_labels.size()[0]).expand_as(distance)
        p1 = prod_labels.view(prod_labels.size()[0], 1).expand_as(distance)
        positives_ids = torch.eq(p0, p1).to(self.device, dtype=torch.float32) - (torch.eye(len(distance))).to(
            self.device)

        n_ids = ((positives_ids > 0) + (distance < self.thr)).to(self.device, dtype=torch.float32)
        unused_neg_part = 0
        if not for_cluster:
            n_ids_all = (distance[positives_ids == 0]).to(self.device, dtype=torch.float32)
            unused_neg_part = (torch.sum(n_ids) / torch.sum(n_ids_all)).item()
            log_metrics('clusters/unused_neg_part_within_cutoff', unused_neg_part, iter_, cfg=self.cfg)
            log_metrics(f'clusters/unused_neg_part_within_cutoff_learner_id_{learner_id}', unused_neg_part,
                        iter_, cfg=self.cfg)

        total_neg_dist = torch.mean(distance[positives_ids == 0])
        negatives_ids = n_ids * 1e6 + distance
        to_retrieve_ids = max(1, min(int(positives_ids.data.sum()) // len(positives_ids), negatives_ids.size(1)))
        negatives = negatives_ids.topk(to_retrieve_ids, dim=1, largest=False)[1]
        negatives_ids_ = torch.zeros_like(negatives_ids.data).scatter(1, negatives, 1.0)

        for i in range(len(distance)):
            anchor_ids.extend([i] * (int(positives_ids.data.sum()) // len(positives_ids)))
            pos_ids_ = np.atleast_1d(positives_ids[i].nonzero().squeeze().cpu().numpy())
            neg_ids_ = np.atleast_1d(negatives_ids_[i].nonzero().squeeze().cpu().numpy())
            pos_ids.extend(pos_ids_)
            neg_ids.extend(neg_ids_)

            if len(anchor_ids) != len(pos_ids) or len(anchor_ids) != len(neg_ids):
                t = min(map(len, [anchor_ids, pos_ids, neg_ids]))
                anchor_ids = anchor_ids[:t]
                pos_ids = pos_ids[:t]
                neg_ids = neg_ids[:t]
        anchors, positives, negatives = embeddings[anchor_ids], embeddings[pos_ids], embeddings[neg_ids]
        return anchor_ids, anchors, positives, negatives, total_neg_dist, unused_neg_part

    def forward(self, embeddings, product_labels, iter_, learner_id=-1, for_cluster=False):
        a_indices, anchors, positives, negatives, total_neg_dist, unused_neg_part = \
            self.sample_triplets(embeddings, product_labels, iter_=iter_, learner_id=learner_id,
                                 for_cluster=for_cluster)

        log_metrics('loss/beta', self.beta.data.tolist()[0], iter_, cfg=self.cfg)
        log_metrics(f'loss/beta_{learner_id}', self.beta.data.tolist()[0], iter_, cfg=self.cfg)

        beta_reg_loss = torch.norm(self.beta, p=1) * self.nu if a_indices is not None else 0.0
        d_ap = torch.sqrt(torch.sum((positives - anchors) ** 2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors) ** 2, dim=1) + 1e-8)
        d_pn = torch.sqrt(torch.sum((positives - negatives) ** 2, dim=1) + 1e-8)
        pos_loss = torch.clamp(d_ap - self.beta[0] + self.margin, min=0.0)
        neg_loss = torch.clamp(self.beta[0] - d_an + self.margin, min=0.0)
        nrof_not_null_loss = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))
        loss = torch.sum(pos_loss + neg_loss) + beta_reg_loss
        if nrof_not_null_loss>0:
            loss = loss/nrof_not_null_loss
        log_metrics('loss/margin_loss', loss.item(), iter_, cfg=self.cfg)
        if not for_cluster:
            return loss, d_ap, d_an, d_pn, torch.sum(pos_loss > 0.0) / float(pos_loss.view(-1).size(0)), \
                   torch.sum(neg_loss > 0.0) / float(neg_loss.view(-1).size(0)), \
                   float(neg_loss.view(-1).size(0)), total_neg_dist, \
                   nrof_not_null_loss, self.beta.data.tolist()[0], unused_neg_part
        else:
            return loss
