from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import SwinForImageClassification
from torchvision import transforms
from scipy.spatial.distance import cosine
import torch
import base64
import os
import io
import asyncio


app = Flask(__name__)
CORS(app)  # 啟用 CORS

# 設定後端照片的目錄
directory = "./set"
imgInfo=[
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

# 選擇 Swin 模型
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = model.to(device)

# 定義圖片預處理步驟
preprocess = transforms.Compose([
    #統一辨識圖片之格式
    transforms.Resize((224, 224)),
    #轉換成torch可以處理的格式
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取圖片的特徵
def extract_features(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    features = outputs.logits.squeeze().cpu().numpy()
    return features

# 透過模型找到向量特徵並回傳
async def compute_image_features():
    features_list = []
    for info in imgInfo:
        image_path = info['path']
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            features = extract_features(image)
            features_list.append({"features": features, "location": info['location'], "path": image_path})
    return features_list

# 非同步的圖片比對功能
@app.route('/upload', methods=['POST'])
async def upload_and_compare():
    print("運行中")
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # 獲取上傳的圖片
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # 提取上傳圖片的特徵
    uploaded_features = extract_features(image)

    # 計算與目錄中圖片的相似度
    similar_imgs = []
    
    # 非同步地計算所有圖片特徵
    image_features = await compute_image_features()
    
    for entry in image_features:
        similarity = 1 - cosine(uploaded_features, entry["features"])
        print("相似度為", similarity)
        
        if similarity > 0.55:  # 設定回傳照片的相似度
            with open(entry["path"], "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                
            # 將 features 和 similarity 轉換為 Python 物件
            similar_imgs.append({
                "similarity": round(float(similarity) * 100, 2),  # 確保 similarity 是 float 類型
                "location": entry["location"],
                "image": encoded_image
            })

    return jsonify(similar_imgs)
# 啟動 Flask
if __name__ == '__main__':
    print("伺服器啟動中，歡迎使用！")
    app.run(debug=True, port=5000)
