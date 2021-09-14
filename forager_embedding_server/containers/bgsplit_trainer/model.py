import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
import torchvision.models.resnet

class ResNetBackbone(vision.models.resnet.ResNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def resnet50(pretrained) -> ResNetBackbone:
    arch = 'resnet50'
    model = ResNetBackbone(vision.models.resnet.Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            vision.models.resnet.model_urls[arch],
            progress=False)
        model.load_state_dict(state_dict)
    return model

class Model(nn.Module):
    def __init__(self, num_main_classes, num_aux_classes,
                 freeze_backbone=False):
        super(Model, self).__init__()
        self.backbone = resnet50(pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        backbone_feature_dim = self.backbone.fc.in_features
        self.main_head = nn.Linear(backbone_feature_dim, num_main_classes)
        self.auxiliary_head = None
        if num_aux_classes:
            self.auxiliary_head = nn.Linear(backbone_feature_dim, num_aux_classes)

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):
        feature = self.forward_backbone(x)
        main_logits = self.main_head(feature)
        if self.auxiliary_head:
            aux_logits = self.auxiliary_head(feature)
            return main_logits, aux_logits
        return main_logits

    def predict(self, x):
        feature = self.forward_backbone(x)
        main_logits = self.main_head(feature)
        return F.softmax(main_logits, dim=1)
