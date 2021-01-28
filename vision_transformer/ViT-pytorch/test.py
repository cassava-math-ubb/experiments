import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from urllib.request import urlretrieve
from glob import glob

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


if __name__ == "__main__":
    config = CONFIGS["ViT-B_16"]
    model = VisionTransformer(config, num_classes=5, zero_head=False, img_size=224, vis=True)
    model.load_state_dict(torch.load("output/cassava_trial1_checkpoint.pth"))
    model.eval()

    labels = pd.read_csv("/home/ligia/Documents/facultate/master-1.1/modelare_numerica/leaf_classification/dataset/train.csv")

    accuracy = 0.0
    count = 0

    for path in glob("/home/ligia/Documents/facultate/master-1.1/modelare_numerica/leaf_classification/dataset/train/*.jpg"):
        img_name = path.split("/")[-1]

        im = Image.open(path)
        x = transform(im)
        x.size()

        logits, att_mat = model(x.unsqueeze(0))
        probs = torch.nn.Softmax(dim=-1)(logits)
        top5 = torch.argsort(probs, dim=-1, descending=True)

        true_label = np.where(labels["image_id"] == img_name)[0][0]

        if top5[0][0] == labels["label"][true_label]:
            accuracy += 1
        count += 1

    accuracy /= count
    
    print(f"Accuracy: {accuracy}")
