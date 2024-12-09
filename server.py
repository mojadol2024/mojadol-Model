import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
import json

# Flask 애플리케이션 초기화
app = Flask(__name__)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의 및 가중치 로드
model = models.densenet201(pretrained=False)
num_classes = 12  # 학습 당시 사용된 클래스 수
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

model_load_path = "./dog_classifier_densenet201_final_2.pth"
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.eval()

print(f"모델이 {model_load_path} 경로에서 성공적으로 로드되었습니다.")

# 클래스 레이블 정의
class_labels = [
    "비숑", "보더콜리", "불독", "치와와", "골든 리트리버",
    "말티즈", "포메라니안", "푸들", "시바견", "시츄", "시베리안 허스키",
    "요크셔 테리어"
]

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class DogClassifier:
    def __init__(self, model_path, class_names, img_size=(224, 224)):
        self.img_size = img_size
        self.class_names = class_names
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return tf.keras.models.load_model(self.model_path)

    def predict(self, img):
        """이미지 예측"""
        img = tf.image.resize(img, self.img_size)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        predictions = self.model.predict(img, verbose=0)  # verbose=0으로 출력 제거
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return {
            "class": self.class_names[predicted_class],
            "confidence": round(confidence * 100, 2)
        }

# 모델 경로 및 클래스 정의
model_path = "./dog_classifier.keras"
class_names = ["not_dog", "dog"]
dog_classifier = None

try:
    dog_classifier = DogClassifier(model_path, class_names)
except Exception as e:
    print(f"Error initializing classifier: {e}")

# 예측 엔드포인트 정의
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return json.dumps({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try: 
        # 이미지 처리
        img = Image.open(file.stream).convert("RGB")
        img_array = np.array(img)

        # 강아지인지 아닌지 판별
        dog_check = dog_classifier.predict(img_array)
        
        if dog_check["class"] != 'not_dog':
            # 이미지 전처리 및 예측
            img_tensor = preprocess(img).unsqueeze(0).to(device)  # 전처리 및 배치 차원 추가

            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top1_prob, top1_classes = torch.topk(probabilities, 1)
            
            # 예측 결과 JSON 응답으로 변환
            response = {
                "Predicted": class_labels[top1_classes[0].item()],
                "confidence": f"{top1_prob[0].item() * 100:.2f}%",
                "is_dog": True
            }
        else:
            # 강아지가 아닌 경우
            response = {
                "Predicted": "not_dog",
                "confidence": "N/A",
                "is_dog": False
            }
        
        return json.dumps(response, ensure_ascii=False), 200
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), 500
    
@app.route('/test', methods=['POST'])
def test():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    try:
        # 이미지 열기
        img = Image.open(file.stream).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # 전처리 및 배치 차원 추가

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_classes = torch.topk(probabilities, 5)


        return class_labels[top5_classes[0].item()], 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
