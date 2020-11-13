from torch import nn
import torch.nn.functional as F
from torchvision import models
from efficientnet_pytorch import EfficientNet


class ResNetTrasferLearning(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


class EfficientNetTransferLearning(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b1')

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model._fc.in_features
        self.model._fc.requires_grad = True
        print(num_ftrs)

        self.model._fc = nn.Linear(num_ftrs, 128)
        self.lin_1 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.model(x)
        x = self.lin_1(x)
        return F.log_softmax(x, dim=1)


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
