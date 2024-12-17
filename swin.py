import os
import torch
from transformers import SwinForImageClassification
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cosine
import numpy as np

# 設定目錄和待比較圖片的完整路徑
directory = "./set"
img_to_compare = "image.png"  # 確保提供完整路徑
[
    {"path": "set\\九份老街\\image (1).jpeg", "location": "九份老街"},
    {"path": "set\\九份老街\\image (1).png", "location": "九份老街"},
    {"path": "set\\九份老街\\image (2).jpeg", "location": "九份老街"},
    {"path": "set\\九份老街\\image.jpeg", "location": "九份老街"},
    {"path": "set\\九份老街\\image.png", "location": "九份老街"},
    {"path": "set\\台北101\\image (1).jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image (1).png", "location": "台北101"},
    {"path": "set\\台北101\\image (2).jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image (3).jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image (4).jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image (5).jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image.jpeg", "location": "台北101"},
    {"path": "set\\台北101\\image.png", "location": "台北101"},
    {"path": "set\\夕陽\\elephant.png", "location": "夕陽"},
    {"path": "set\\夕陽\\elephant2.png", "location": "夕陽"},
    {"path": "set\\夕陽\\image (1).jpeg", "location": "夕陽"},
    {"path": "set\\夕陽\\image (2).jpeg", "location": "夕陽"},
    {"path": "set\\夕陽\\image (3).jpeg", "location": "夕陽"},
    {"path": "set\\夕陽\\image (4).jpeg", "location": "夕陽"},
    {"path": "set\\夕陽\\image.jpeg", "location": "夕陽"},
    {"path": "set\\夕陽\\懸日.png", "location": "夕陽"},
    {"path": "set\\夕陽\\類日懸日.png", "location": "夕陽"},
    {"path": "set\\森林\\image (1).jpeg", "location": "森林"},
    {"path": "set\\森林\\image (2).jpeg", "location": "森林"},
    {"path": "set\\森林\\image (3).jpeg", "location": "森林"},
    {"path": "set\\森林\\image (4).jpeg", "location": "森林"},
    {"path": "set\\森林\\image.jpeg", "location": "森林"},
    {"path": "set\\海岸\\image (1).jpeg", "location": "海岸"},
    {"path": "set\\海岸\\image (2).jpeg", "location": "海岸"},
    {"path": "set\\海岸\\image (3).jpeg", "location": "海岸"},
    {"path": "set\\海岸\\image (4).jpeg", "location": "海岸"},
    {"path": "set\\海岸\\image.jpeg", "location": "海岸"},
    {"path": "set\\海岸\\image.png", "location": "海岸"},
    {"path": "set\\海岸\\斷牙.jpg", "location": "海岸"},
    {"path": "set\\海岸\\斷牙2.jpg", "location": "海岸"},
    {"path": "set\\港口\\image (1).jpeg", "location": "港口"},
    {"path": "set\\港口\\image (2).jpeg", "location": "港口"},
    {"path": "set\\港口\\image.jpeg", "location": "港口"}
]


# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 加載 Swin 模型並將其移動到 GPU
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = model.to(device)  # 將模型移動到 GPU 或 CPU
model.eval()  # 設置為推理模式

# 定義圖片預處理步驟
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取圖片的特徵
def extract_features(image_path):
    try:
        # 加載圖片
        image = Image.open(image_path).convert("RGB")  # 確保圖片是 RGB 模式
        # 使用 torchvision 的 transforms 進行預處理
        image = preprocess(image).unsqueeze(0)  # 增加 batch 維度
        image = image.to(device)  # 將圖片移動到 GPU 或 CPU

        # 通過模型進行預測，獲得 logits
        with torch.no_grad():
            outputs = model(image)
        
        # 提取 logits 作為特徵
        features = outputs.logits.squeeze().cpu().numpy()  # 將特徵移回 CPU 並轉換為 numpy 數組
        return features
    except Exception as e:
        print(f"處理圖片 {image_path} 時出現錯誤: {e}")
        return None

# 獲取目錄中的所有圖片
def get_images(base_dir):
    imgs = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # 只選擇圖片文件
                imgs.append(os.path.join(root, file))
    return imgs

# 計算並顯示每張圖片的相似度
image_paths = get_images(directory)

# 提取待比較圖片的特徵
features_to_compare = extract_features(img_to_compare)
if features_to_compare is None:
    print("無法提取待比較圖片的特徵，程序結束。")
    exit()

for img_path in image_paths:
    if img_path != img_to_compare:
        features1 = extract_features(img_path)  # 提取當前圖片的特徵
        if features1 is not None:
            # 計算餘弦相似度
            similarity = 1 - cosine(features1, features_to_compare)
            if similarity * 100 > 55:
                print(f"{img_to_compare} 跟 {img_path} 的相似度為 {similarity * 100:.2f}%")
print("完成")
