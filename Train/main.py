import numpy as np
import time
import torch
import os
import tarfile
import gc
from warmup_scheduler import GradualWarmupScheduler
from apex import amp
from tensorboardX import SummaryWriter

from Configs.baseline_config import cfg
from Utils.train_utils import log_metrics, make_training_step, make_density_plots, get_optimizer, get_criterion
from Losses.LabelSmoothClassificationLoss import LabelSmoothLoss
from Utils.eval_utils import evaluate, evaluate_isc
from Utils.clustering_utils import crazy_stuff
from Models.learners_model import get_model

from Datasets.CARSDataset import CARSDataset
from Datasets.CUBDataset import CUBDataset
from Datasets.InShopDataset import InShopDataset
from Datasets.OnlineProductsDataset import OnlineProductsDataset


def train(model, criterion, optimizer):
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.cur_dataset is 'OnlineProducts':
        ds_train = OnlineProductsDataset(ds_path=cfg.sop_dataset_path, mode='train')
    elif cfg.cur_dataset is 'InShopClothes':
        ds_train = InShopDataset(ds_path=cfg.inshop_dataset_path, mode='train')
    elif cfg.cur_dataset is 'CUB200':
        ds_train = CUBDataset(ds_path=cfg.cub_dataset_path, mode='train')
    elif cfg.cur_dataset is 'CARS196':
        ds_train = CARSDataset(ds_path=cfg.cars_dataset_path, mode='train')
    else:
        raise Exception
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.batch_size)

    dl_train_list, cluster_labels, product_labels, indexes = None, None, None, None
    is_fine_tune_epoch, is_fine_tune_epoch1 = False, False
    start_epoch, global_step = 0, -1

    # if use_mean_beta_at_finetune:
    #     criterion[-1].beta.data = torch.tensor([torch.mean(
    #         torch.tensor([crit.beta.data for i, crit in enumerate(criterion[:-1])]))]).cuda()

    if cfg.use_lr_scheduler:
        num_steps = 200000/cfg.nb_clusters
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-7)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100,
                                                  after_scheduler=lr_scheduler)

    if cfg.continue_training_from_epoch:
        try:
            checkpoint = torch.load(os.path.join(cfg.input_dir, f'checkpoint_{cfg.checkpoint_from_epoch}.pth'))
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step'] + 1
            optimizer.load_state_dict(checkpoint['opt'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(cfg.device)
            if cfg.use_amp:
                amp.load_state_dict(checkpoint['amp_state_dict'])
            for i, crit in enumerate(criterion):
                crit.beta.data = checkpoint['beta'][i]
            print(f'Loading model saved in epoch: {start_epoch - 1}')
        except FileNotFoundError:
            print('Checkpoint not found')

        print('Trying to load saved clusters..')
        try:
            clusters_checkpoint = torch.load(
                os.path.join(cfg.input_dir, f'clusters_at_epoch_{cfg.clusters_from_epoch}.pth'))
            cl_epoch = clusters_checkpoint['epoch'] + 1
            dl_train_list = clusters_checkpoint['dl_train_list']
            cluster_labels = clusters_checkpoint['cluster_labels']
            indexes = clusters_checkpoint['indexes']
            centroids = clusters_checkpoint['centroids']
            print(f'Loading clusters saved in epoch: {cl_epoch - 1}')
        except FileNotFoundError:
            print('Clusters checkpoint not found')

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    cross_entropy = LabelSmoothLoss(0.1) if cfg.use_cls_loss else None
    optimizer.zero_grad()
    optimizer.step()
    if cfg.use_lr_scheduler:
        lr_scheduler.step(global_step // cfg.nb_clusters)
        warmup_scheduler.step(global_step // cfg.nb_clusters)

    for param_group in optimizer.param_groups:
        log_metrics('loss/lr', param_group['lr'], global_step, cfg=cfg)
        continue
    if cfg.compute_metrics_before_training:
        model.eval()
        assert not model.training
        with torch.no_grad():
            if cfg.cur_dataset is 'InShopClothes':
                metrics = evaluate_isc(None, -1, model_, dl_query, dl_gallery, cfg=cfg)
            else:
                metrics = evaluate(None, -1, dl_=dl_test, model_=model_, cfg=cfg)
        model.train()
        assert model.training

    for e in range(start_epoch, start_epoch + cfg.epochs):
        if e >= cfg.ft_epoch:
            is_fine_tune_epoch = True
        if e >= cfg.fine_tune_epoch:
            is_fine_tune_epoch1 = True
        print(f'Epoch: {e}')
        epoch_start_time = time.time()
        writer = SummaryWriter(log_dir= cfg.tensorboard_dir + f'/epoch_{e}')
        # writer = None

        if e % cfg.recluster_epoch_freq == 0:
            model.eval()
            assert not model.training
            print('Clustering..')
            with torch.no_grad():
                if e == 0:
                    dl_train_list, cluster_labels, product_labels, indexes, centroids = \
                        crazy_stuff(model, dl_train, writer, e, criterion[-1], cfg=cfg, is_fine_tune=is_fine_tune_epoch,
                                    is_fine_tune1=is_fine_tune_epoch1)
                else:
                    dl_train_list, cluster_labels, product_labels, indexes, centroids = \
                        crazy_stuff(model, dl_train, writer, e, criterion[-1], cfg=cfg, is_reclustering_epoch=True,
                                    previous_indexes=indexes,
                                    previous_cluster_labels=cluster_labels, previous_centroids=centroids,
                                    is_fine_tune=is_fine_tune_epoch, is_fine_tune1=is_fine_tune_epoch1)
            for cluster_id in range(cfg.nb_clusters):
                if writer is not None:
                    writer.add_scalar('nb_images/cluster_%d' % cluster_id, np.sum(cluster_labels == cluster_id), e)
                log_metrics('clusters/nb_images_cluster_%d' % cluster_id, np.sum(cluster_labels == cluster_id), e,
                            cfg=cfg)
            if cfg.save_clusters:
                print('Saving clusters...')
                clusters_checkpoint = {
                    'epoch': e,
                    'dl_train_list': dl_train_list,
                    'cluster_labels': cluster_labels,
                    'product_labels': product_labels,
                    'indexes': indexes,
                    'centroids': centroids
                }
                torch.save(clusters_checkpoint,
                           (os.path.join(cfg.checkpoints_dir, f'clusters_at_epoch_{e}.pth')))

        model.train()
        assert model.training
        print('Starting training..')
        nb_batches = [len(dl) for dl in dl_train_list]
        print('nb_bathes in each dl', nb_batches)
        for nb_b in nb_batches:
            log_metrics('clusters/nb_batches', nb_b, e, cfg=cfg)
        if cfg.cur_dataset in ['CUB200', 'CARS196']:
            _nb_batches_ = min(nb_batches)
        else:
            _nb_batches_ = int(np.mean(nb_batches))  # max(nb_batches)
        loss_list, d_ap_list, d_an_list, d_pn_list, beta_list, unused_neg_part_list = \
            list(), list(), list(), list(), list(), list()
        loss_by_learners, d_ap_by_learners, d_an_by_learners, d_pn_by_learners, beta_by_learners = \
            {i: [] for i in range(cfg.nb_clusters)}, {i: [] for i in range(cfg.nb_clusters)}, \
            {i: [] for i in range(cfg.nb_clusters)}, {i: [] for i in range(cfg.nb_clusters)}, \
            {i: [] for i in range(cfg.nb_clusters)}
        unused_neg_part_by_learners = {i: [] for i in range(cfg.nb_clusters)}
        nb_iters = _nb_batches_ * cfg.nb_clusters
        print(f'Iterations num: {nb_iters}')
        log_metrics('clusters/Iterations num', nb_iters, e, cfg=cfg)
        with open(os.path.join(cfg.metrics_dir, f'model.masks'), 'a') as f:
            f.write(f'epoch_{e} ' + str(model.masks) + '\n')

        gc.collect()
        for j in range(_nb_batches_):
            for i, dl in enumerate(dl_train_list):
                if len(dl) != 0:
                    dl = iter(dl)
                    batch = next(dl)
                    if batch is None:
                        dl = iter(dl)
                        batch = next(dl)
                    learner_id = i
                    loss_id = -1 if is_fine_tune_epoch else i

                    with torch.autograd.set_detect_anomaly(True):
                        loss_on_batch, d_ap, d_an, d_pn, non_zero_part_pos, non_zero_part_neg, \
                        nrof_triplets, total_neg_dist, nrof_not_null_loss, beta, unused_neg_part = make_training_step(
                            batch, learner_id, model, criterion[loss_id], optimizer,
                            global_step, e, cfg=cfg, cross_entropy=cross_entropy,
                            is_fine_tune=is_fine_tune_epoch)
                        if loss_on_batch is not None:
                            global_step += 1
                            #                         if writer is not None:
                            log_metrics('loss/Loss_train', loss_on_batch, global_step, cfg=cfg)
                            log_metrics(f'loss/_loss_learner_{learner_id}', loss_on_batch, global_step, cfg=cfg)
                            log_metrics(f'distances/d_ap_learner_{learner_id}', d_ap.mean().item(), global_step,
                                        cfg=cfg)
                            log_metrics(f'distances/d_an_learner_{learner_id}', d_an.mean().item(), global_step,
                                        cfg=cfg)
                            log_metrics(f'distances/d_pn_learner_{learner_id}', d_pn.mean().item(), global_step,
                                        cfg=cfg)

                            log_metrics('loss/non_zero_part_pos_loss', non_zero_part_pos, global_step, cfg=cfg)
                            log_metrics(f'loss/non_zero_part_pos_loss_learner_{learner_id}', loss_on_batch, global_step,
                                        cfg=cfg)

                            log_metrics('loss/non_zero_part_neg_loss', non_zero_part_neg, global_step, cfg=cfg)
                            log_metrics(f'loss/non_zero_part_neg_loss_learner_{learner_id}', non_zero_part_neg,
                                        global_step, cfg=cfg)

                            log_metrics('loss/nrof_not_null_loss', nrof_not_null_loss, global_step, cfg=cfg)
                            log_metrics(f'loss/nrof_not_null_loss_learner_{learner_id}', nrof_not_null_loss,
                                        global_step, cfg=cfg)

                            log_metrics(f'clusters/nrof_triplets_learner_{learner_id}', nrof_triplets, global_step,
                                        cfg=cfg)
                            log_metrics(f'clusters/nrof_triplets_all', nrof_triplets, global_step, cfg=cfg)
                            log_metrics(f'distances/total_neg_dist_learner_{learner_id}', total_neg_dist.item(),
                                        global_step, cfg=cfg)
                            log_metrics(f'distances/total_neg_dist_all', total_neg_dist.item(), global_step, cfg=cfg)
                            for crit_i in range(len(criterion_)):
                                log_metrics(f'loss/beta_learner_id_{crit_i}', criterion_[crit_i].beta[0].item(),
                                            global_step, cfg=cfg)

                            loss_list.append(loss_on_batch)
                            loss_by_learners[learner_id].append(loss_on_batch)
                            d_ap_by_learners[learner_id].append(d_ap.data.cpu().numpy().tolist())
                            d_an_by_learners[learner_id].append(d_an.data.cpu().numpy().tolist())
                            d_pn_by_learners[learner_id].append(d_pn.data.cpu().numpy().tolist())
                            unused_neg_part_by_learners[learner_id].append(unused_neg_part)
                            beta_by_learners[learner_id].append(beta)

                            d_ap_list.append(d_ap.data.cpu().numpy().tolist())
                            d_an_list.append(d_an.data.cpu().numpy().tolist())
                            d_pn_list.append(d_pn.data.cpu().numpy().tolist())
                            beta_list.append(beta)
                            unused_neg_part_list.append(unused_neg_part)

                            if global_step % 50 == 0:
                                if global_step != 0:
                                    loss_mean = np.mean(loss_list[-50:])
                                else:
                                    loss_mean = loss_on_batch
                                print(f'global step: {global_step}, loss: {loss_mean}')

            if cfg.use_lr_scheduler:
                lr_scheduler.step(global_step//cfg.nb_clusters)
                warmup_scheduler.step(global_step//cfg.nb_clusters)
                param_gr = ['backbone', 'embedding', 'masks', 'beta']
                for p_, param_group in enumerate(optimizer.param_groups):
                    if writer is not None:
                        writer.add_scalar(f'lr_{param_gr[p_]}', param_group['lr'], global_step)
                    log_metrics(f'loss/lr_{param_gr[p_]}', param_group['lr'], global_step, cfg)
                    continue

        log_metrics('distances/d_ap_mean', np.mean(d_ap_list), e, cfg=cfg)
        log_metrics('distances/d_an_mean', np.mean(d_an_list), e, cfg=cfg)
        log_metrics('distances/d_pn_mean', np.mean(d_pn_list), e, cfg=cfg)
        log_metrics('loss/mean_loss', np.mean(loss_list), e, cfg=cfg)
        log_metrics('loss/mean_beta', np.mean(beta_list), e, cfg=cfg)
        log_metrics('clusters/mean_unused_w_cutoff_neg_part', np.mean(unused_neg_part_list), e, cfg=cfg)

        for learner_id in range(cfg.nb_clusters):
            log_metrics('loss/mean_loss_learner_id_%d' % learner_id, np.mean(loss_by_learners[learner_id]), e, cfg=cfg)
            log_metrics('distances/mean_d_ap_learner_id_%d' % learner_id, np.mean(d_ap_by_learners[learner_id]), e,
                        cfg=cfg)
            log_metrics('distances/mean_d_an_learner_id_%d' % learner_id, np.mean(d_an_by_learners[learner_id]), e,
                        cfg=cfg)
            log_metrics('distances/mean_d_pn_learner_id_%d' % learner_id, np.mean(d_pn_by_learners[learner_id]), e,
                        cfg=cfg)
            log_metrics('loss/mean_beta_learner_id_%d' % learner_id, np.mean(beta_by_learners[learner_id]), e, cfg=cfg)
            log_metrics('clusters/mean_unused_w_cutoff_neg_part_learner_id_%d' % learner_id,
                        np.mean(unused_neg_part_by_learners[learner_id]), e, cfg=cfg)

        if cfg.to_make_density_plots and e % 2 == 0 and not is_fine_tune_epoch1:
            print(f'making density plots..')
            d_ap_list_ = np.concatenate(d_ap_list, 0)
            d_an_list_ = np.concatenate(d_an_list, 0)
            make_density_plots(e, d_ap_list_, d_an_list_, cfg, mode='train')

        print(f'epoch training time: {round((time.time() - epoch_start_time) / 60, 3)} min')
        eval_start_time = time.time()
        model.eval()
        assert not model.training
        with torch.no_grad():
            if cfg.cur_dataset is 'InShopClothes':
                metrics = evaluate_isc(writer, e, model_, dl_query, dl_gallery, cfg)
            else:
                metrics = evaluate(writer, e, dl_=dl_test, model_=model_, cfg=cfg)
                if cfg.evalaute_on_train_data:
                    metrics = evaluate(writer, e, dl_=dl_train, model_=model_, cfg=cfg, mode='train')

        if cfg.use_mean_beta_at_finetune and e == cfg.ft_epoch - 1:
            criterion[-1].beta.data = torch.tensor(
                [torch.mean(torch.tensor([crit.beta.data for i, crit in enumerate(criterion[:-1])]))]).cuda()
        #             print('beta at finetune start', criterion[-1].beta.data)

        if cfg.save_model:
            print('Saving current model...')
            state = {
                'model': model.state_dict(),
                'amp_state_dict': amp.state_dict() if cfg.use_amp else None,
                'epoch': e,
                'global_step': global_step,
                'opt': optimizer.state_dict(),
                'beta': [criterion_[i].beta for i in range(len(criterion_))]
            }
            torch.save(state, (os.path.join(cfg.checkpoints_dir, f'checkpoint_{e}.pth')))

        # files_to_save = os.listdir('/kaggle/working/')
        # tar = tarfile.open(f'mlruns_epoch_{e}.tar.gz', 'w:gz')
        # tar1 = tarfile.open(f'all_metrics_{e}.tar.gz', 'w:gz')
        # for item in files_to_save:
        #     if item.startswith('mlruns') and not item.startswith('mlruns_epoch'):
        #         tar.add(item)
        #     if item in cfg.dirs_for_logs:
        #         tar1.add(item)
        # tar.close()
        # tar1.close()

        #         save_all_data_to_tar(f'all_epoch_{e}', e)

        # if os.path.exists(os.path.join(cfg.checkpoints_dir, f'checkpoint_{e - 5}.pth')):
        #     os.remove(os.path.join(cfg.checkpoints_dir, f'checkpoint_{e - 5}.pth'))
        #
        # if os.path.exists(os.path.join(cfg.checkpoints_dir, f'clusters_at_epoch_{e - 10}.pth')):
        #     os.remove(os.path.join(cfg.checkpoints_dir, f'clusters_at_epoch_{e - 10}.pth'))
        #
        # if os.path.exists(os.path.join(cfg.checkpoints_dir, f'mlruns_epoch_{e - 5}.tar.gz')):
        #     os.remove(os.path.join(cfg.checkpoints_dir, f'mlruns_epoch_{e - 5}.tar.gz'))

        #         if os.path.exists(os.path.join(checkpoints_dir, f'all_epoch_{e-2}.tar.gz')):
        #             os.remove(os.path.join(checkpoints_dir, f'all_epoch_{e-2}.tar.gz'))

        print(f'validation time: {round((time.time() - eval_start_time) / 60, 3)} min')
        print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')


def test(model, epoch):
    print(f'testing model..')
    if cfg.load_saved_model_for_test:
        try:
            checkpoint = torch.load(os.path.join(cfg.input_dir, f'checkpoint_{epoch}.pth'))
            model.load_state_dict(checkpoint['model'])
            cur_epoch = checkpoint['epoch']
            print(f'Loading model saved in epoch: {cur_epoch}')
        except FileNotFoundError:
            print('Checkpoint not found')
    model.eval()
    assert not model.training
    with torch.no_grad():
        if cfg.cur_dataset is 'InShopClothes':
            metrics = evaluate_isc(None, -1, model_, dl_query, dl_gallery, cfg)
        else:
            metrics = evaluate(None, -1, dl_=dl_test, model_=model_, cfg=cfg)


if __name__ == '__main__':
    print(f'selected dataset: {cfg.cur_dataset}')
    torch.cuda.set_device(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # memory = get_gpu_memory()

    if cfg.cur_dataset is 'OnlineProducts':
        train_classes, test_classes = cfg.sop_train_classes, cfg.sop_test_classes
        ds_test = OnlineProductsDataset(ds_path=cfg.sop_dataset_path, mode='test')
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.batch_size)
        nb_train_data = cfg.sop_nb_train_data

    elif cfg.cur_dataset is 'InShopClothes':
        train_classes, test_classes = cfg.inshop_train_classes, cfg.inshop_test_classes
        ds_query = InShopDataset(ds_path=cfg.inshop_dataset_path, mode='query')
        ds_gallery = InShopDataset(ds_path=cfg.inshop_dataset_path, mode='gallery')
        dl_query = torch.utils.data.DataLoader(ds_query, batch_size=cfg.batch_size)
        dl_gallery = torch.utils.data.DataLoader(ds_gallery, batch_size=cfg.batch_size)
        nb_train_data = cfg.inshop_nb_train_data

    elif cfg.cur_dataset is 'CUB200':
        train_classes, test_classes = cfg.cub_train_classes, cfg.cub_test_classes
        ds_test = CUBDataset(ds_path=cfg.cub_dataset_path, mode='test')
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.batch_size)
        nb_train_data = cfg.cub_nb_train_data

    elif cfg.cur_dataset is 'CARS196':
        train_classes, test_classes = cfg.cars_train_classes, cfg.cars_test_classes
        ds_test = CARSDataset(ds_path=cfg.cars_dataset_path, mode='test')
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.batch_size)
        nb_train_data = cfg.cars_nb_train_data

    model_ = get_model(cfg).cuda()
    criterion_ = get_criterion(cfg)
    all_betas = [criterion_[i].beta for i in range(len(criterion_))]
    optimizer_ = get_optimizer(model_, all_betas, cfg)
    if cfg.use_amp:
        model_, optimizer_ = amp.initialize(model_, optimizer_, opt_level='O1')

    #     test(model_, 186)

    total_training_start_time = time.time()
    train(model_, criterion_, optimizer_)
    print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')

    # test(model_, 0)
