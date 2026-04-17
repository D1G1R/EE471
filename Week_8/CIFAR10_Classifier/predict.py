import os
os.environ["TORCH_HOME"] = "."

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from cog import BasePredictor, Input, Path
from PIL import Image

# 1. Senin Colab'de Eğittiğin Baseline Model Mimarisi
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

class Predictor(BasePredictor):
    def setup(self):
        """Modeli belleğe yükle"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        
        # Optimize edilmemiş modelimizi yüklüyoruz
        self.model.load_state_dict(torch.load("cifar_net.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # CIFAR10 Sınıfları
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']

    def predict(self, image: Path = Input(description="Image to classify")) -> dict:
        """Tahmin işlemini çalıştır"""
        # CIFAR10 renkli olduğu için RGB formatında açıyoruz (Invert'e gerek yok)
        img = Image.open(image).convert("RGB")
        
        # Eğitimdeki transformasyonların aynısını uyguluyoruz
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # [Batch, Channel, H, W] formatı
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(img_tensor)
            probs = F.softmax(preds[0], dim=0) # F.softmax kullanıyoruz
            top3_prob, top3_indices = probs.topk(3)
            
        return {self.classes[i]: p.detach().item() for p, i in zip(top3_prob, top3_indices)}