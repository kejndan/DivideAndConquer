import torch
import numpy as np
import json
import seaborn as sns
from matplotlib import pyplot as plt
import torch.nn.functional as F
import faiss
import tarfile
from apex import amp

from BatchSampler.BatchSampler import BatchSampler
from Datasets.CARSDataset import CARSDataset
from Datasets.CUBDataset import CUBDataset
from Datasets.InShopDataset import InShopDataset
from Datasets.OnlineProductsDataset import OnlineProductsDataset
from Losses.MarginLoss import MarginLoss
from Utils.log_utils import *




def make_training_step(batch, learner_id, model, criterion, optimizer, iter_, epoch, cfg, cross_entropy=None,
                       is_fine_tune=False):
    images, product_labels = batch[0].cuda(), batch[1].cuda()
    optimizer.zero_grad()

    if cross_entropy is not None:
        embeddings, logits = model(images, with_softmax=True)
        cls_loss = cross_entropy(logits, product_labels) * 0.02
        log_metrics('loss/cls_loss', cls_loss.item(), iter_, cfg=cfg)
    else:
        embeddings = model(images)
        cls_loss = 0

    if not is_fine_tune:
        current_mask = model.masks(torch.tensor(learner_id).cuda())
        current_mask = torch.nn.functional.relu(current_mask)
        l1_regularization = cfg.lambda1 * torch.norm(current_mask, p=1) / embeddings.size(0)
        l1_regularization = l1_regularization.cuda()
        log_metrics('masks/nrof_positive_mask_%d ' % learner_id, torch.nonzero(current_mask).size(0), iter_, cfg=cfg)
        log_metrics('masks/mean_mask_%d ' % learner_id, current_mask.mean().item(), iter_, cfg=cfg)
        l2_regularization = cfg.lambda2 * embeddings.norm(2) / np.sqrt(embeddings.size(0))
        embeddings = embeddings * current_mask
    else:
        l1_regularization = 0
        l2_regularization = 0

    l2_regularization = l2_regularization + cfg.lambda2 * embeddings.norm(2) / np.sqrt(embeddings.size(0))
    l2_regularization = l2_regularization.cuda()
    # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    loss, d_ap, d_an, d_pn, non_zero_part_pos, non_zero_part_neg, nrof_triplets, total_neg_dist, \
    nrof_not_null_loss, beta, unused_neg_part = criterion(embeddings, product_labels, iter_, learner_id=learner_id)

    loss = loss + l1_regularization + l2_regularization + cls_loss
    if not is_fine_tune:
        log_metrics('loss/L1_masks_reg', l1_regularization.item(), iter_, cfg=cfg)
        log_metrics('loss/L1_masks_reg_learner_id_%d ' % learner_id, l1_regularization.item(), iter_, cfg=cfg)
    log_metrics('loss/L2_emb_reg', l2_regularization.item(), iter_, cfg=cfg)
    log_metrics('loss/L2_emb_reg_learner_id_%d ' % learner_id, l2_regularization.item(), iter_, cfg=cfg)
    log_metrics('loss/total_loss', loss.item(), iter_, cfg=cfg)
    log_metrics('distances/d_ap', d_ap.mean().item(), iter_, cfg=cfg)
    log_metrics('distances/d_ap_learner_id_%d ' % learner_id, d_ap.mean().item(), iter_, cfg=cfg)
    log_metrics('distances/d_an', d_an.mean().item(), iter_, cfg=cfg)
    log_metrics('distances/d_an_learner_id_%d ' % learner_id, d_an.mean().item(), iter_, cfg=cfg)
    if not torch.isnan(loss).any():
        if cfg.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        with open(f'{cfg.metrics_dir}masks_epoch_{epoch}.json', 'w') as out_file:
            masks_ = model.masks.weight.cpu().detach().numpy().tolist()
            json.dump({'masks': masks_}, out_file)
        log_artifacts(f'{cfg.metrics_dir}masks_epoch_{epoch}.json', f'masks_epoch_{epoch}.json', cfg=cfg)

        return loss.item(), d_ap, d_an, d_pn, non_zero_part_pos.item(), non_zero_part_neg.item(), nrof_triplets, \
               total_neg_dist, nrof_not_null_loss, beta, unused_neg_part
    else:
        print(f'Loss is nan')
        return None, d_ap, d_an, d_pn, non_zero_part_pos.item(), non_zero_part_neg.item(), nrof_triplets, \
               total_neg_dist, nrof_not_null_loss, beta, unused_neg_part


def get_train_dl(cfg, idxs=None):
    if cfg.cur_dataset is 'OnlineProducts':
        ds_train_ = OnlineProductsDataset(ds_path=cfg.sop_dataset_path, mode='train', transforms_mode='train')
    elif cfg.cur_dataset is 'InShopClothes':
        ds_train_ = InShopDataset(ds_path=cfg.inshop_dataset_path, mode='train', transforms_mode='train')
    elif cfg.cur_dataset is 'CUB200':
        ds_train_ = CUBDataset(ds_path=cfg.cub_dataset_path, mode='train', transforms_mode='train')
    elif cfg.cur_dataset is 'CARS196':
        ds_train_ = CARSDataset(ds_path=cfg.cars_dataset_path, mode='train', transforms_mode='train')
    ds_train_.get_within_indexes(idxs)
    dl_train_ = torch.utils.data.DataLoader(ds_train_, batch_sampler=BatchSampler(ds_train_, cfg.batch_size))
    return dl_train_


def compute_embeddings(_model, dataloader, cfg, with_embedding_layer=True, with_norm=True):
    print('Computing embeddings..')
    start_time = time.time()
    _model.eval()
    assert not _model.training

    embeddings_set = [[] for _ in range(3)]
    all_embeddings = []
    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            if b % 100 == 0:
                print(f'batch {b}/{len(dataloader)}')
            for ind, elems in enumerate(batch):
                if ind == 0:
                    if not with_embedding_layer:
                        elems, embeddings = _model(elems.to(cfg.device), with_embedding_layer)
                        all_embeddings.extend(embeddings.data.cpu().numpy())
                    else:
                        elems = _model(elems.to(cfg.device), with_embedding_layer)
                    if with_norm:
                        elems = F.normalize(elems, p=2, dim=1)
                    elems = elems.data.cpu().numpy()
                for elem in elems:
                    embeddings_set[ind].append(np.asarray(elem))
        result = [np.stack(embeddings_set[i]) for i in range(len(embeddings_set))]  # 3 x (samples_num x 2048)
    print(f'computing embeddings time: {round((time.time() - start_time) / 60, 3)}, min')
    if not with_embedding_layer:
        return result, all_embeddings
    return result


def make_density_plots(epoch, pos_distances, neg_distances, cfg, save_fig=True,
                       show=False, mode='valid', log_distances=False):
    sns.set(color_codes=True)
    sns.kdeplot(pos_distances, shade=True, color='r', label='anchor-positive')
    sns.kdeplot(neg_distances, shade=True, color='b', label='anchor-negative')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    if save_fig:
        plt.savefig(os.path.join(cfg.density_plots_dir, mode + f'_density_plot_at_epoch_{epoch}'))
        if log_distances:
            json.dump(list(pos_distances),
                      open(os.path.join(cfg.density_plots_dir, 'a_p_dists_' + mode + f'_epoch_{epoch}'), 'w'))
            json.dump(list(neg_distances),
                      open(os.path.join(cfg.density_plots_dir, 'a_n_dists_' + mode + f'_epoch_{epoch}'), 'w'))
        log_artifacts(os.path.join(cfg.density_plots_dir, mode + f'_density_plot_at_epoch_{epoch}.png'),
                      mode + f'_density_plot_at_epoch_{epoch}.png', cfg=cfg)
    if show:
        plt.show()
    plt.clf()


def get_criterion(cfg):
    _criterion = [MarginLoss(cfg).cuda() for _ in range(cfg.nb_clusters + 1)]
    return _criterion


def get_optimizer(_model_, beta, cfg):
    opt_ = torch.optim.Adam([
        {'params': _model_.parameters_dict['backbone'], 'lr': cfg.lr},  # 'weight_decay': weight_decay},
        {'params': _model_.parameters_dict['embedding'], 'lr': cfg.lr},  # 'weight_decay': weight_decay},
        {'params': _model_.parameters_dict['masks'], 'lr': cfg.masks_lr},
        {'params': beta, 'lr': 0.01, 'weight_decay': 0.0001}])
    return opt_


def save_all_data_to_tar(tarfile_name, cur_epoch):
    try:
        all_files = os.listdir('/kaggle/working/')
        tar = tarfile.open(f"{tarfile_name}.tar.gz", "w:gz")
        for item in all_files:
            if not item.startswith(
                    'all') and f'_{cur_epoch}.' in item or f'_{cur_epoch - 1}.' in item or item.startswith('metrics'):
                tar.add(item)
        tar.close()
    except:
        print('error while writing to tarfile')


# def get_gpu_memory():
#     gpu_resources = faiss.StandardGpuResources()
#     idx_config = faiss.GpuIndexFlatConfig()
#     idx_config.useFloat16 = False
#     idx_config.device = 0
#     index = faiss.GpuIndexFlatL2(gpu_resources, 2048, idx_config)
#     return index, gpu_resources
