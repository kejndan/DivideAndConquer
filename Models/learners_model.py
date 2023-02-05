# from math import ceil
import numpy as np
# import torchvision
import torch
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d, Module
from torch.nn.init import xavier_normal_
from Models.resnet_model import get_resnet50
from torch.nn import functional as f




def mac(x):
    return torch.flatten(f.max_pool2d(x, (x.size(-2), x.size(-1))), 1)
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return torch.flatten(f.avg_pool2d(x, (x.size(-2), x.size(-1))), 1)
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


class MAC(Module):
    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(Module):
    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def gem(x, p=3, eps=1e-6):
    return torch.flatten(f.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p), 1)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


class GeM(Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


def get_params_dict(model, emb_module_name=['embedding', 'masks']):
    dict_ = {k: [] for k in ['backbone', *emb_module_name]}
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if name not in emb_module_name:
            dict_['backbone'] += [param]
        else:
            dict_[name] += [param]
    nb_total = len(list(model.parameters()))
    nb_dict_params = sum([len(dict_[d]) for d in dict_])
    assert nb_total == nb_dict_params
    return dict_


def get_embedding(model, cfg):
    if cfg.use_gem:
        model.features_pooling = GeM().to(cfg.device)
    else:
        model.features_pooling = AvgPool2d(7, stride=1, padding=0, ceil_mode=True, count_include_pad=True)
    model.features_dropout = Dropout(0.01)
    torch.random.manual_seed(1)
    model.embedding = Linear(model.sz_features_output, cfg.embedding_dim).to(list(model.parameters())[0].device)
    if cfg.use_cls_loss:
        model.classification_layer = Linear(model.sz_features_output, len(cfg.train_classes)).to(cfg.device)
    model.masks = torch.nn.Embedding(cfg.nb_clusters, cfg.embedding_dim)
    # initialize weights
    model.masks.weight.data.normal_(0.9, 0.7)  # 0.1, 0.005
    torch.random.manual_seed(1)
    np.random.seed(1)
    # _xavier_init
    model.embedding.weight.data = xavier_normal_(model.embedding.weight.data, gain=1)
    features_parameters = model.features.parameters()
    model.parameters_dict = get_params_dict(model=model)

    def forward(x, with_embedding_layer=True, learner_id=None, with_softmax=False):
        x = model.features(x)
        x = model.features_pooling(x)
        x = model.features_dropout(x)
        features = x.view(x.size(0), -1)
        if with_embedding_layer:
            x = model.embedding(features)
            if with_softmax:
                return x, model.classification_layer(features)
            return x
        else:
            embeddings = model.embedding(features)
            return features, embeddings

    model.forward = forward


def get_model(cfg):
    resnet50 = get_resnet50()
    get_embedding(resnet50, cfg)
    return resnet50
