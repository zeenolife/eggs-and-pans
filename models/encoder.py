from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url


class ResNet50Encoder(ResNet):

    def __init__(self, pretrained=True):

        # Inheriting ResNet 50 as is
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3])

        # Loading pretrained ImageNet weights
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            state_dict = load_state_dict_from_url(model_url)
            self.load_state_dict(state_dict)

        # Delete last layers
        del self.fc
        del self.avgpool

    def forward(self, x):

        # Conv 1 of ResNet
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        # Conv 2 of ResNet
        conv2 = self.maxpool(conv1)
        conv2 = self.layer1(conv2)

        # Conv 3 of ResNet
        conv3 = self.layer2(conv2)

        # Conv 4 of ResNet
        conv4 = self.layer3(conv3)

        # Conv 5 of ResNet
        conv5 = self.layer4(conv4)

        return [conv1, conv2, conv3, conv4, conv5]
