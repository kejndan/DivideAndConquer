import torchvision
import torch


def get_resnet50():
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.features = torch.nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool,
                                            resnet50.layer1,
                                            resnet50.layer2, resnet50.layer3, resnet50.layer4)
    resnet50.sz_features_output = 2048
    for module in filter(lambda m: type(m) == torch.nn.BatchNorm2d, resnet50.modules()):
        module.eval()
        module.train = lambda _: None
    return resnet50
