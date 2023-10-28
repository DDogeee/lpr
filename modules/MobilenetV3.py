import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from PIL import Image

class MobileNetV3():
    def __init__(self, pretrained = None, device = 'cpu'):
        model = models.mobilenet_v3_large(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, 35)
        if pretrained is not None:
            model.load_state_dict(torch.load(pretrained, map_location=torch.device(device)))
        model.classifier = model.classifier[0]
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.classes = ['2', '0', 'E', 'N', 'Y', 'A', 'Z', '4', 'T', 'Q', 'H', 'B', 'W', 'F', 'G', '6', 'L', 'X', 'J', '5', 'R','D', 'U', 'I', '8', 'C', '7', 'V', '1', '9', 'S', 'P', 'K', 'M', '3']
        self.model.eval()
    def __call__(self, image = None, image_path = None):
        if image_path != None:
            image = Image.open(image_path)
        else:
            image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0)
        return self.model(image)
