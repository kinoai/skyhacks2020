import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet

from sky_utils.yolo_names import yolo_to_sky_mapper, class_names


class EfficientNetTransferLearning(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        # self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # self.model = EfficientNet.from_pretrained('efficientnet-b7')

        for param in self.model.parameters():
            param.requires_grad = True

        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, config["lin1_size"])
        self.lin_1 = nn.Linear(config["lin1_size"], config["lin2_size"])
        self.lin_2 = nn.Linear(config["lin2_size"], config["output_size"])

    def forward(self, x):
        x = self.model(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.lin_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2)
        x = self.lin_2(x)
        return torch.sigmoid(x)


class ResNetTrasferLearning(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, config["lin1_size"]),
            nn.ReLU(),
            nn.Linear(config["lin1_size"], config["output_size"]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MNISTExampleModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = nn.Linear(config["num_of_inputs"], config["lin1_size"])
        self.layer_2 = nn.Linear(config["lin1_size"], config["lin2_size"])
        self.layer_3 = nn.Linear(config["lin2_size"], config["lin3_size"])
        self.layer_4 = nn.Linear(config["lin3_size"], 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # for mnist: (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)
        x = self.layer_4(x)

        return F.log_softmax(x, dim=1)


class YoloModel(nn.Module):
    def __init__(self, probability_treshold=0.8):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()  # yolov5s.pt
        self.model = self.model.autoshape()
        self.probability_treshold = probability_treshold

    def forward(self, x):
        predictions = self.model(x)
        y = torch.zeros(38)

        if len(predictions) > 0 and predictions[0] is not None:
            for pred in predictions[0]:
                class_index = int(pred[5].item())
                if class_index in yolo_to_sky_mapper.keys() and pred[4].item() > self.probability_treshold:
                    y[yolo_to_sky_mapper[class_index]] = 1

        return torch.Tensor(y)
