from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.nn as nn


class MaskRCNN(nn.Module):
    """ Neural network """

    def __init__(self, num_classes=2, hidden_size=256):
        super().__init__()
        # load an instance segmentation model pre-trained pre-trained on COCO
        self.model_ft = maskrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = self.model_ft.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = self.model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
        # and replace the mask predictor with a new one
        self.model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_size, num_classes
        )

    def forward(self, images, targets):
        """ Forward pass for your feedback prediction network. """
        out = self.model_ft(images, targets)
        return out

    def forward_with_activations(self, x):
        pass
