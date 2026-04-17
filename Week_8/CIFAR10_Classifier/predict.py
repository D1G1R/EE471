import os
os.environ["TORCH_HOME"] = "."

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from cog import BasePredictor, Input, Path
from PIL import Image

# 1. Baseline Model Mimarisi
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Optimize Edilmiş Model Mimarisi (Dropout %30)
class OptimizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']

        # 1. Baseline Modeli Yükle
        self.model_base = Net()
        self.model_base.load_state_dict(torch.load("cifar_net.pth", map_location=self.device))
        self.model_base.to(self.device).eval()

        # 2. Optimize Modeli Yükle
        self.model_opt = OptimizedNet()
        self.model_opt.load_state_dict(torch.load("cifar_net_optimized.pth", map_location=self.device))
        self.model_opt.to(self.device).eval()

    def predict(self, image: Path = Input(description="Image to classify")) -> dict:
        img = Image.open(image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_base = self.model_base(img_tensor)
            pred_opt = self.model_opt(img_tensor)
            
            # Sınıf indekslerini al
            class_base = self.classes[pred_base[0].argmax(0).item()]
            class_opt = self.classes[pred_opt[0].argmax(0).item()]
            
        return {
            "baseline_prediction": class_base,
            "optimized_prediction": class_opt
        }