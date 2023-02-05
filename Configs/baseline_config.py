from easydict import EasyDict

cfg = EasyDict()
cfg.sop_dataset_path = '/kaggle/input/sopdataset/Stanford_Online_Products/'
cfg.inshop_dataset_path = '/kaggle/input/inshopclothes/In_shop Clothes Retrieval Benchmark'
cfg.cub_dataset_path = '/kaggle/input/cub200/CUB_200_2011'
cfg.cars_dataset_path = '/kaggle/input/cars196'
cfg.pretrained_pytorch_models = '/kaggle/input/pretrained-pytorch-Models'
cfg.input_dir = '/kaggle/input/checkpoints'

cfg.checkpoints_dir = '/kaggle/working/'
cfg.tensorboard_dir = '/kaggle/working/'
cfg.losses_dir = '/kaggle/working/'
cfg.metrics_dir = '/kaggle/working/'
cfg.density_plots_dir = '/kaggle/working/'

cfg.device = 'cuda:0'
cfg.random_seed = 0
cfg.lr = 5e-5
cfg.embedding_dim = 128
cfg.batch_size = 80
cfg.masks_lr = 5e-5

# OnlineProducts params
cfg.sop_train_classes = range(0, 11318)
cfg.sop_test_classes = range(11318, 22634)
cfg.sop_nb_train_data = 59551

# InShop params
cfg.inshop_train_classes = range(0, 3997)
cfg.inshop_test_classes = range(0, 3985)
cfg.inshop_nb_train_data = 25882

# CARS196 params
cfg.cars_train_classes = range(1, 99)
cfg.cars_test_classes = range(99, 197)
cfg.cars_nb_train_data = 8054

# CUB200 params
cfg.cub_train_classes = range(1, 101)
cfg.cub_test_classes = range(101, 201)
cfg.cub_nb_train_data = 5864

cfg.epochs = 250
cfg.fine_tune_epoch = 250  # 190
cfg.recluster_epoch_freq = 2

cfg.load_saved_model_for_test = True
cfg.save_clusters = True
cfg.save_model = True
cfg.to_make_density_plots = True
cfg.clustering_by_embeddings=False
cfg.use_amp = False

cfg.continue_training_from_epoch = False
cfg.checkpoint_from_epoch = None
cfg.clusters_from_epoch = None

cfg.load_all_metrics = False  # True if continue training
cfg.all_metrics_dir = '' #f'/kaggle/input/checkpoints/all_metrics_{cfg.checkpoint_from_epoch}'

# to eval model with loaded checkpoint or to check model performance before training
cfg.compute_metrics_before_training = False
cfg.evalaute_on_train_data = False
cfg.lambda1 = 5e-4  # masks l1 regularization
cfg.lambda2 = 5e-3

cfg.available_datasets = [
    'OnlineProducts',
    'InShopClothes',
    'CUB200',
    'CARS196'
]
cfg.cur_dataset = 'OnlineProducts'
cfg.use_gem = True
cfg.use_lr_scheduler = True
cfg.use_mean_beta_at_finetune = True  # False
cfg.use_cls_loss = False
if cfg.use_cls_loss:
    if cfg.cur_dataset == 'OnlineProducts':
        cfg.train_classes = cfg.sop_train_classes
    elif cfg.cur_dataset == 'InShopClothes':
        cfg.train_classes = cfg.inshop_train_classes
    elif cfg.cur_dataset == 'CUB200':
        cfg.train_classes = cfg.cub_train_classes
    elif cfg.cur_dataset == 'CARS196':
        cfg.train_classes = cfg.cars_train_classes

if cfg.cur_dataset in ['CUB200', 'CARS196']:
    cfg.nb_clusters = 4
else:
    cfg.nb_clusters = 8

cfg.ft_epoch = 60 if cfg.cur_dataset is 'CUB200' or cfg.cur_dataset is 'CARS196' else 80
