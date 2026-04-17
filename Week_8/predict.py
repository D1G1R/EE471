import os
os.environ["TORCH_HOME"] = "."

import torch
from torch import nn
from torchvision import transforms
from cog import BasePredictor, Input, Path
from PIL import Image, ImageOps

# 1. Baseline Model Mimarisi
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))

# 2. Optimize Edilmiş Model Mimarisi (Dropout'lu)
class OptimizedNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))

class Predictor(BasePredictor):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sınıf İsimleri
        self.classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

        # 1. Baseline Modeli Yükle
        self.model_base = NeuralNetwork()
        self.model_base.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model_base.to(self.device).eval()

        # 2. Optimize Modeli Yükle
        self.model_opt = OptimizedNeuralNetwork()
        self.model_opt.load_state_dict(torch.load("optimized_model.pth", map_location=self.device))
        self.model_opt.to(self.device).eval()

    # ESKİ PREDICT METODU (YORUM)
    # def predict(self, image: Path = Input(description="Image to classify")) -> dict:
    #     img = Image.open(image).convert("L")
    #     transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    #     img_tensor = transform(img).unsqueeze(0).to(self.device)
    #     
    #     with torch.no_grad():
    #         pred_base = self.model_base(img_tensor)[0]
    #         pred_opt = self.model_opt(img_tensor)[0]
    #         class_base = self.classes[pred_base.argmax(0).item()]
    #         class_opt = self.classes[pred_opt.argmax(0).item()]
    #     
    #     return {
    #         "baseline_prediction": class_base,
    #         "optimized_prediction": class_opt
    #     }

    def predict(self, image: Path = Input(description="Image to classify")) -> dict:
        # Resmi siyah-beyaz (L) formatında aç
        img = Image.open(image).convert("L")
        
        # 🛠️ KRİTİK DÜZELTME: Renkleri tersine çevir! 
        # (Beyaz arka planı siyaha, siyah kıyafeti beyaza çevirir)
        img = ImageOps.invert(img)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_base = self.model_base(img_tensor)[0]
            pred_opt = self.model_opt(img_tensor)[0]
            
            class_base = self.classes[pred_base.argmax(0).item()]
            class_opt = self.classes[pred_opt.argmax(0).item()]
            
        return {
            "baseline_prediction": class_base,
            "optimized_prediction": class_opt
        }