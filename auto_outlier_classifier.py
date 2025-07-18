import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # FC 제거, 2048차원 임베딩만
model.eval()
model.to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

root_dir = 'resources/only_crawl'
outlier_dir = 'resources/flagged_outlier'
os.makedirs(outlier_dir, exist_ok=True)

threshold = 2.0  # 표준편차

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img).squeeze().cpu().numpy()
    return feat


for class_name in tqdm(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path): continue

    img_paths = [os.path.join(class_path, fname)
                 for fname in os.listdir(class_path)
                 if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]

    feats = []
    valid_paths = []
    for p in img_paths:
        try:
            feats.append(get_embedding(p))
            valid_paths.append(p)
        except Exception:
            pass

    feats = np.stack(feats)
    mean_vec = np.mean(feats, axis=0)
    dists = np.linalg.norm(feats - mean_vec, axis=1)
    std = np.std(dists)
    mean_dist = np.mean(dists)

    for idx, dist in enumerate(dists):
        if dist > mean_dist + threshold * std:
            out_path = os.path.join(outlier_dir, f"{class_name}_{os.path.basename(valid_paths[idx])}")
            shutil.move(valid_paths[idx], out_path)

    if np.mean(dists) > 1.5 * np.median(dists):
        print(f"[경고] 클래스 '{class_name}' 내부 이미지들이 서로 다를 수 있음!")

print("이상치 탐지/이동 완료!")
