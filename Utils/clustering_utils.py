import numpy as np
import faiss
import gc
import time
import torch
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_
import json

from Utils.train_utils import compute_embeddings, get_train_dl
from Utils.log_utils import log_metrics

def kmeans(x, num_clusters, previous_centroids=None, nrns_clustering=False):
    x = np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    x = np.ascontiguousarray(x)
    dim_ = x.shape[1]
    kmeans_model = faiss.Clustering(x.shape[1], num_clusters)
    # initializing clusters with previous centroids
    if previous_centroids is not None:
        print(f'clustering with previous centroids..')
        kmeans_model.centroids.resize(previous_centroids.size)
        faiss.memcpy(kmeans_model.centroids.data(), faiss.swig_ptr(previous_centroids), previous_centroids.size * 4)
    kmeans_model.max_points_per_centroid = 10000000
    kmeans_model.niter = 100
    if previous_centroids is None:
        kmeans_model.nredo = 5
    else:
        kmeans_model.nredo = 1
    resources_ = faiss.StandardGpuResources()
    idx_config_ = faiss.GpuIndexFlatConfig()
    idx_config_.useFloat16 = False
    idx_config_.device = 0
    index = faiss.GpuIndexFlatL2(resources_, dim_, idx_config_)
    kmeans_model.train(x, index)
    centroids = faiss.vector_float_to_array(kmeans_model.centroids)
    # objective = faiss.vector_float_to_array(kmeans_model.obj)
    centroids = centroids.reshape(num_clusters, dim_)
    index.reset()
    del index
    del idx_config_
    gc.collect()
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)
    distances, labels = index.search(x, 1)
    index.reset()
    resources_.noTempMemory()
    del resources_
    del kmeans_model
    del index
    del x
    gc.collect()
    return labels.ravel(), centroids


def get_nearest_neighbors(x, product_labels, k, queries=None):
    x = np.asarray(x.reshape(x.shape[0], -1), dtype=np.float32)
    with_queries = False
    if queries is None:
        with_queries = True
        queries = x
        k += 1
    dim_ = x.shape[1]
    resources_ = faiss.StandardGpuResources()
    idx_config_ = faiss.GpuIndexFlatConfig()
    idx_config_.useFloat16 = False
    idx_config_.device = 0
    index = faiss.GpuIndexFlatL2(resources_, dim_, idx_config_)
    index.add(x)
    _, neighbors = index.search(queries, k)
    if with_queries:
        for i in range(len(neighbors)):
            indices = np.nonzero(neighbors[i, :] != i)[0]
            indices.sort()
            if len(indices) > k - 1:
                indices = indices[:-1]
            neighbors[i, :-1] = neighbors[i, indices]
        neighbors = neighbors[:, :-1]
    n_neighbors = np.array([[product_labels[i] for i in ii] for ii in neighbors])
    index.reset()
    resources_.noTempMemory()
    del index
    del x
    del resources_
    del neighbors
    del idx_config_
    gc.collect()
    return n_neighbors


def cluster_embeddings(_model, dl, cfg, previous_centroids=None, is_finetune=False):
    if is_finetune:
        prod_labels = np.array(dl.dataset.labels)
        emb_indexes = np.array(dl.dataset.ids)
        cluster_labels_ = np.zeros(len(prod_labels), dtype=int)
        centroids, embeddings, final_embeddings = None, None, None
    else:
        (embeddings, prod_labels, emb_indexes), final_embeddings = compute_embeddings(_model, dl, cfg,
                                                                                      with_embedding_layer=cfg.clustering_by_embeddings,
                                                                                      with_norm=True)  # True
        print(f'emb shapes while clustering: {embeddings.shape}')
        sorted_lists = sort_(emb_indexes, [emb_indexes, embeddings, prod_labels])
        emb_indexes, embeddings, prod_labels = sorted_lists[0], sorted_lists[1], sorted_lists[2]
        start_time = time.time()
        cluster_labels_, centroids = kmeans(
            embeddings,
            num_clusters=cfg.nb_clusters,
            previous_centroids=previous_centroids)

        print(f'clustering time: {round((time.time() - start_time) / 60, 3)} min')
    return cluster_labels_, prod_labels, emb_indexes, centroids, embeddings, final_embeddings


def get_optimal_nb_clusters_by_loss(embeddings, prod_labels, previous_centroids, writer, epoch,
                                    criterion, tensorboard_dir):
    start_time = time.time()
    losses = []
    distances_ap = []
    distances_an = []
    non_zeros_pos = []
    non_zeros_neg = []
    nb_clusters_list = []
    total_nrof_triplets = []
    total_neg_distances = []
    print('start validate nb_clusters...')
    for nb_clusters_ in range(8, 40):
        cluster_labels_, centroids = kmeans(
            embeddings,
            num_clusters=nb_clusters_,
            previous_centroids=previous_centroids)
        print(nb_clusters_)
        nb_clusters_list.append(nb_clusters_)
        # print(f'distance: {distance}')
        # writer.add_scalar('distance_%d'%nb_clusters_, distance, epoch)
        current_loss = []
        current_ap = []
        current_an = []
        current_non_zero_part_pos = []
        current_non_zero_part_neg = []
        current_nrof_triplets = []
        current_nrof_not_null_loss = []
        current_neg_dist = []
        for cluster_id in np.unique(cluster_labels_):
            indices = np.arange(len(embeddings), dtype=np.int)[cluster_labels_ == cluster_id]
            loss, d_ap, d_an, d_pn, non_zero_part_pos, non_zero_part_neg, nrof_triplets, \
            total_neg_dist, nrof_not_null_loss = criterion(
                torch.tensor(embeddings[indices]).cuda(),
                torch.tensor(prod_labels[indices]).cuda(), epoch)
            if np.isnan(loss.item()):
                continue
            current_loss.append(loss.item())
            current_ap.append(d_ap.mean().item())
            current_an.append(d_an.mean().item())
            current_non_zero_part_pos.append(non_zero_part_pos.item())
            current_non_zero_part_neg.append(non_zero_part_neg.item())
            current_nrof_triplets.append(nrof_triplets)
            current_neg_dist.append(total_neg_dist.item())
            current_nrof_not_null_loss.append(nrof_not_null_loss)
            print(current_loss[-1], current_ap[-1], current_an[-1])
        losses.append(np.mean(current_loss))
        distances_ap.append(np.mean(current_ap))
        distances_an.append(np.mean(current_an))
        non_zeros_pos.append(np.mean(current_non_zero_part_pos))
        non_zeros_neg.append(np.mean(current_non_zero_part_neg))
        total_nrof_triplets.append(np.mean(current_nrof_triplets))
        total_neg_distances.append(np.mean(current_neg_dist))
        writer.add_scalar('loss/epoch_%d' % epoch, np.mean(current_loss), nb_clusters_)
        writer.add_scalar('d_ap/epoch_%d' % epoch, np.mean(current_ap), nb_clusters_)
        writer.add_scalar('d_an/epoch_%d' % epoch, np.mean(current_an), nb_clusters_)
        writer.add_scalar('non_zero_part_pos/epoch_%d' % epoch, np.mean(current_non_zero_part_pos), nb_clusters_)
        writer.add_scalar('non_zero_part_neg/epoch_%d' % epoch, np.mean(current_non_zero_part_neg), nb_clusters_)
        writer.add_scalar('nrof_triplets/epoch_%d' % epoch, np.mean(current_nrof_triplets), nb_clusters_)
        writer.add_scalar('current_neg_dist/epoch_%d' % epoch, np.mean(current_neg_dist), nb_clusters_)
        writer.add_scalar('current_nrof_not_null_loss/epoch_%d' % epoch, np.mean(current_nrof_not_null_loss),
                          nb_clusters_)
    plt.cla()
    plt.plot(nb_clusters_list, losses, label='losses', color='r')
    plt.plot(nb_clusters_list, non_zeros_pos, label='non_zeros_pos', color='b')
    plt.plot(nb_clusters_list, non_zeros_neg, label='non_zeros_neg', color='g')
    # plt.plot(nb_clusters_list, total_neg_distances, label='total_neg_distances', color='y')
    plt.legend()
    plt.savefig(tensorboard_dir + 'different_nb_clusters_%d.png' % epoch)
    plt.cla()
    plt.plot(nb_clusters_list, total_nrof_triplets, label='total_nrof_triplets', color='b')
    plt.savefig(tensorboard_dir + 'total_nrof_triplets_different_nb_clusters_%d.png' % epoch)
    # plt.show()
    plt.cla()
    print(f'clustering time: {round((time.time() - start_time) / 60, 3)} min')
    best_index = int(np.argmax(losses))
    nb_clusters_ = nb_clusters_list[best_index]
    gc.collect()
    print('optimal nb_clusters: %d, loss: %.3f, d_ap: %.3f, d_an: %.3f' % (nb_clusters_, losses[best_index],
                                                                           distances_ap[best_index],
                                                                           distances_an[best_index]))


def sort_(indexes_, lists_to_sort):
    sorted_indexes = np.sort(indexes_)
    sorted_lists = [list_[sorted_indexes] for list_ in lists_to_sort]
    return sorted_lists


def round_neurons_num(neurons_percents, embedding_dim):
    def err(p, rounded_p):
        d = np.sqrt(1.0 if p < 1.0 else p)
        return abs(rounded_p - p) ** 2 / d

    if not np.isclose(sum(neurons_percents), embedding_dim):
        print(f'neurons_percents: {neurons_percents}')
        raise ValueError
    n = len(neurons_percents)
    rounded = [int(x) for x in neurons_percents]
    dif = embedding_dim - sum(rounded)
    errs = [(err(neurons_percents[i], rounded[i] + 1) - err(neurons_percents[i], rounded[i]), i) for i in range(n)]
    r = sorted(errs)
    for i in range(dif):
        rounded[r[i][1]] += 1
    return rounded


def crazy_stuff(_model_,
                dataloader_train,
                writer,
                epoch,
                criterion, cfg,
                is_reclustering_epoch=False,
                previous_indexes=None,
                previous_cluster_labels=None,
                previous_centroids=None,
                is_fine_tune=False,
                is_fine_tune1=False):
    if is_reclustering_epoch:
        cluster_labels_, product_labels_, indexes_, centroids, embeddings, final_embeddings = cluster_embeddings(
            _model_,
            dataloader_train,
            cfg,
            previous_centroids=previous_centroids,
            is_finetune=is_fine_tune1)
    else:
        cluster_labels_, product_labels_, indexes_, centroids, embeddings, final_embeddings = cluster_embeddings(
            _model_,
            dataloader_train,
            cfg,
            is_finetune=is_fine_tune1)
    #     with torch.no_grad():
    #         get_optimal_nb_clusters_by_loss(np.asarray(final_embeddings), product_labels_, previous_centroids,
    #         writer, epoch, criterion)
    sorted_list = sort_(indexes_, [indexes_, product_labels_, cluster_labels_, embeddings])
    indexes_, product_labels_, cluster_labels_, embeddings = \
        sorted_list[0], sorted_list[1], sorted_list[2], sorted_list[3]

    nrof_reassign_images = 0
    for c in range(cfg.nb_clusters):
        for t in np.unique(product_labels_[cluster_labels_ == c]):
            condition = product_labels_[cluster_labels_ == c] == t
            if np.sum(condition).item() == 1:
                if np.sum(product_labels_ == t).sum().item() == 1:
                    cluster_labels_[(product_labels_ == t) & (cluster_labels_ == c)] = -1
                    continue
                nrof_reassign_images += 1
                embedding = embeddings[cluster_labels_ == c, :][condition, :]
                dist_to_clusters = np.linalg.norm(embedding - centroids, axis=1)
                product_clusters = np.unique(cluster_labels_[product_labels_ == t])
                dist_to_clusters[c] = 10e7
                dist_to_clusters[np.logical_not(np.isin(np.arange(cfg.nb_clusters), product_clusters))] = 10e7
                assert np.any(dist_to_clusters < 10e6)
                new_c = np.argmin(dist_to_clusters)
                cluster_labels_[(product_labels_ == t) & (cluster_labels_ == c)] = new_c

    log_metrics('clusters/nrof_unused_images', np.sum(cluster_labels_ == -1), epoch, cfg=cfg)

    # if not (is_fine_tune or is_fine_tune1) and epoch%10 == 0:
    #     calculate_neurons_nmi(cluster_labels_, product_labels_, final_embeddings, epoch)

    # get NMI and loss of clusters
    current_loss = []
    t1 = time.time()
    with torch.no_grad():
        for c in np.unique(cluster_labels_):
            indices = np.arange(len(final_embeddings), dtype=np.int)[cluster_labels_ == c]
            emb = torch.tensor([final_embeddings[ind] for ind in indices]).cuda()
            prod_labels = torch.tensor([product_labels_[ind] for ind in indices]).cuda()
            loss = criterion(emb, prod_labels, epoch, for_cluster=True)
            if np.isnan(loss.item()):
                continue
            current_loss.append(loss.item())
            emb = emb.cpu().numpy()
            prod_labels = prod_labels.cpu().numpy()
            cl_lbls, _ = kmeans(emb, num_clusters=len(np.unique(prod_labels)))
            cl_nmi = nmi_(cl_lbls, prod_labels) * 100
            log_metrics(f'clusters/cluster_{c}_nmi', cl_nmi, epoch, cfg=cfg)
            log_metrics(f'clusters/cluster_{c}_loss', np.mean(current_loss), epoch, cfg=cfg)
            print(f'cluster: {c}, nmi: {cl_nmi}, loss: {loss}')
    print(f'getting loss and nmi for clusters time: {(time.time() - t1) / 60} min')

    # get num_of_imgs_in_dif_clusters
    nb_imgs_in_other_clusters = []
    for i, pr_lbl in enumerate(np.unique(product_labels_)):
        clusters_ids = cluster_labels_[product_labels_ == pr_lbl]
        cnt = Counter(clusters_ids)
        most_common_cluster, nb_imgs = cnt.most_common(1)[0]
        count_in_other_clusters = np.sum(list(cnt.values())) - nb_imgs
        nb_imgs_in_other_clusters.append(count_in_other_clusters)
    part_imgs_from_other_clusters = np.sum(nb_imgs_in_other_clusters) / len(product_labels_)
    print(f'part_imgs_from_other_clusters: {part_imgs_from_other_clusters}')
    log_metrics(f'clusters/part_imgs_from_other_clusters1', part_imgs_from_other_clusters, epoch, cfg=cfg)

    dls = [[] for _ in range(cfg.nb_clusters)]
    for cluster_id in range(cfg.nb_clusters):
        idxs = indexes_[cluster_labels_ == cluster_id]
        dls[cluster_id] = get_train_dl(cfg, idxs=idxs)
    return dls, cluster_labels_, product_labels_, indexes_, centroids



def calculate_neurons_nmi(cluster_labels_, product_labels_, embeddings_, epoch, cfg):
    print(f'assigning neurons with nmi + lsa...')
    # st = time.time()
    result = {}
    embeddings_ = np.asarray(embeddings_)
    for cluster_id in range(cfg.nb_clusters):
        result[cluster_id] = {}
        ids = np.where(cluster_labels_ == np.repeat(cluster_id, len(cluster_labels_)))[0]
        # cl_labels = cluster_labels_[ids]
        prod_labels = product_labels_[ids]
        prod_labels_count = len(np.unique(prod_labels))
        embeds = embeddings_[ids]
        assert len(embeds) == len(prod_labels)

        nrn_cluster_lbls, _ = kmeans(embeds, num_clusters=prod_labels_count, nrns_clustering=True)
        assert len(nrn_cluster_lbls) == len(prod_labels)
        nmi = nmi_(prod_labels, nrn_cluster_lbls)
        result[cluster_id]['all_embedding'] = nmi
        result[cluster_id]['exclude'] = {}
        for i in range(cfg.embedding_dim):
            # assert len(nrn_outs) == len(prod_labels)
            nrn_outs = embeds[:, np.arange(cfg.embedding_dim) != i]
            assert len(nrn_outs[0]) == (cfg.embedding_dim - 1)
            nrn_cluster_lbls, _ = kmeans(nrn_outs, num_clusters=prod_labels_count, nrns_clustering=True)
            assert len(nrn_cluster_lbls) == len(prod_labels)
            nmi = nmi_(prod_labels, nrn_cluster_lbls)
            result[cluster_id]['exclude'][i] = nmi
        gc.collect()
    with open(cfg.tensorboard_dir + 'neurons_nmis_%d.json' % epoch, 'w') as file:
        json.dump(result, file)
    del result


def init_learners_masks(cluster_labels_, product_labels_, embeddings_, cfg):
    print(f'initializing learners masks...')
    st = time.time()
    neurons_nmis = np.zeros((cfg.embedding_dim, cfg.nb_clusters))
    for cluster_id in range(cfg.nb_clusters):
        ids = np.where(cluster_labels_ == np.repeat(cluster_id, len(cluster_labels_)))[0]
        # cl_labels = cluster_labels_[ids]
        prod_labels = product_labels_[ids]
        prod_labels_count = len(np.unique(prod_labels))
        embeds = embeddings_[ids]
        assert len(embeds) == len(prod_labels)
        neurons_outs = np.split(embeds, cfg.embedding_dim, axis=1)

        for i, nrn_outs in enumerate(neurons_outs):
            assert len(nrn_outs) == len(prod_labels)
            nrn_cluster_lbls, _ = kmeans(nrn_outs, num_clusters=prod_labels_count, nrns_clustering=True)
            assert len(nrn_cluster_lbls) == len(prod_labels)
            nmi = nmi_(prod_labels, nrn_cluster_lbls)
            neurons_nmis[i][cluster_id] = nmi
    print(f'masks calculating time: {round((time.time() - st) / 60, 3)} min')
    return neurons_nmis