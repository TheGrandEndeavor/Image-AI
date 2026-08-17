"""This program will learn to process images into numerical data."""


import torch
from torchvision import transforms
from PIL import Image


data = Image.open("./images/pure_white.png").convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

x_data = preprocess(data)
print(x_data)
