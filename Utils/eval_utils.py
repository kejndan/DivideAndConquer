import numpy as np
import torch
import os
import gc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_

from Utils.clustering_utils import get_nearest_neighbors, kmeans
from Utils.train_utils import compute_embeddings
from Utils.log_utils import log_metrics


def get_recall_at_k(product_labels, neighbours_ids, cur_k):
    sum_ = 0
    for query_lbl, nearest_ids in zip(product_labels, neighbours_ids):
        if query_lbl in nearest_ids[:cur_k]:
            sum_ += 1
    recall_at_k = sum_ / len(product_labels)
    return recall_at_k


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def eval_func_isc(query_embeddings, gallery_embeddings, query_product_labels, gallery_product_labels, k_list,
                  nb_classes, writer, epoch, with_norm, use_mask, cfg):
    if with_norm:
        query_embeddings = normalized(np.copy(query_embeddings))
        gallery_embeddings = normalized(np.copy(gallery_embeddings))
    neighbours = get_nearest_neighbors(gallery_embeddings, gallery_product_labels, max(k_list),
                                       queries=query_embeddings)
    product_labels_all = np.concatenate([query_product_labels, gallery_product_labels])
    embeddings_all = np.concatenate([query_embeddings, gallery_embeddings])
    clusters_labels, _ = kmeans(embeddings_all, num_clusters=nb_classes)
    nmi = nmi_(clusters_labels, product_labels_all) * 100
    print(f'nmi: {nmi}')
    out = []
    gc.collect()
    torch.cuda.empty_cache()
    for k in k_list:
        recall_at_k = get_recall_at_k(query_product_labels, neighbours, k) * 100
        recall_metrics_name = f'Recall_at_{k}_w_norm' if with_norm else f'Recall_at_{k}_not_norm'
        print(f'{recall_metrics_name}: {recall_at_k}')
        if writer is not None:
            writer.add_scalar(recall_metrics_name, recall_at_k, epoch)
        log_metrics(f'eval/{recall_metrics_name}', recall_at_k, epoch, cfg=cfg)
        out.append(recall_at_k)
    out.append(nmi)
    nmi_metrics_name = 'NMI_w_norm' if with_norm else 'NMI_not_norm'
    if writer is not None:
        writer.add_scalar(nmi_metrics_name, nmi, epoch)
    log_metrics(f'eval/{nmi_metrics_name}', nmi, epoch, cfg=cfg)
    str_ = ' '.join([str(elem) for elem in out])
    with open(os.path.join(cfg.metrics_dir, f'metrics'), 'a') as f:
        f.write(f'epoch_{epoch} norm {with_norm} ' + str_ + '\n')
    return out


def evaluate_isc(writer, epoch, model_, dl_query, dl_gallery, cfg, is_fine_tune=False, with_norm=True,
                 mode=None):  # for InShopClothes dataset
    query_embeddings, query_product_labels, _ = compute_embeddings(model_, dl_query, cfg=cfg, with_norm=False)
    gallery_embeddings, gallery_product_labels, _ = compute_embeddings(model_, dl_gallery, cfg=cfg, with_norm=False)

    print(f'emb shape while evaluating: {query_embeddings.shape}, {gallery_embeddings.shape}')
    nb_classes = dl_query.dataset.get_nb_classes()
    assert nb_classes == len(set(query_product_labels))
    original_query_embeddings = np.copy(query_embeddings)
    original_gallery_embeddings = np.copy(gallery_embeddings)
    k_list = [1, 10, 20, 30, 50]
    for use_mask in ['not']:  # + list(np.arange(8)):
        for with_norm in [False, True]:
            if use_mask == 'not':
                query_embeddings = original_query_embeddings
                gallery_embeddings = original_gallery_embeddings
            # else:
            #     query_embeddings = np.copy(original_query_embeddings) * \
            #                        torch.nn.functional.relu(model_.masks.weight[use_mask]).detach().cpu().numpy()
            #     gallery_embeddings = np.copy(original_gallery_embeddings) * \
            #                          torch.nn.functional.relu(model_.masks.weight[use_mask]).detach().cpu().numpy()

            out = eval_func_isc(query_embeddings, gallery_embeddings, query_product_labels, gallery_product_labels,
                                k_list, nb_classes, writer, epoch, with_norm, use_mask, cfg=cfg)
        gc.collect()
    return out


def eval_func(embeddings, product_labels, k_list, nb_classes, writer, epoch, with_norm, use_mask, cfg):
    if with_norm:
        embeddings = normalized(np.copy(embeddings))
    gc.collect()
    torch.cuda.empty_cache()
    neighbours = get_nearest_neighbors(embeddings, product_labels, max(k_list))
    clusters_labels, _ = kmeans(embeddings, num_clusters=nb_classes)
    nmi = nmi_(clusters_labels, product_labels) * 100
    print(f'nmi: {nmi}')
    out = []
    gc.collect()
    torch.cuda.empty_cache()
    for k in k_list:
        recall_at_k = get_recall_at_k(product_labels, neighbours, k) * 100
        recall_metrics_name = f'Recall_at_{k}_w_norm' if with_norm else f'Recall_at_{k}_not_norm'
        print(f'{recall_metrics_name}: {recall_at_k}')
        if writer is not None:
            writer.add_scalar(recall_metrics_name, recall_at_k, epoch)
        log_metrics(f'eval/{recall_metrics_name}', recall_at_k, epoch, cfg=cfg)
        out.append(recall_at_k)
    out.append(nmi)
    nmi_metrics_name = 'NMI_w_norm' if with_norm else 'NMI_not_norm'
    if writer is not None:
        writer.add_scalar(nmi_metrics_name, nmi, epoch)
    log_metrics(f'eval/{nmi_metrics_name}', nmi, epoch, cfg=cfg)
    str_ = ' '.join([str(elem) for elem in out])
    with open(os.path.join(cfg.metrics_dir, f'metrics'), 'a') as f:
        f.write(f'epoch_{epoch} norm {with_norm} ' + str_ + '\n')
    gc.collect()
    torch.cuda.empty_cache()
    return out


def evaluate(writer, epoch, dl_, model_, cfg, is_fine_tune=False, with_norm=True, mode='test'):
    # compute query images embeddings = retrieval images embeddings
    print(f'Evaluating on {mode} data')
    embeddings, product_labels, _ = compute_embeddings(model_, dl_, with_norm=False, cfg=cfg)
    print(f'emb shape while evaluating: {embeddings.shape}')
    nb_classes = dl_.dataset.get_nb_classes()
    original_embeddings = np.copy(embeddings)
    if cfg.cur_dataset in ['CUB200', 'CARS196']:
        k_list = [1, 2, 4, 8]
    elif cfg.cur_dataset is 'OnlineProducts':
        k_list = [1, 10, 100, 1000]
    else:
        raise Exception

    for use_mask in ['not']:  # + list(np.arange(8)):
        for with_norm in [False, True]:
            if use_mask == 'not':
                embeddings = original_embeddings
            # else:
            #     embeddings = np.copy(original_embeddings) * \
            #                  torch.nn.functional.relu(model_.masks.weight[use_mask]).detach().cpu().numpy()
            out = eval_func(embeddings, product_labels, k_list, nb_classes, writer, epoch, with_norm, use_mask, cfg=cfg)
        gc.collect()
    return out
