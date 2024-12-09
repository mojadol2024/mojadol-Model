import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import gc

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 폰트 경로 설정 (Windows의 경우 "맑은 고딕" 사용)
font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 모델 구조 정의 (출력 클래스 수를 저장된 모델과 동일하게 설정)
model = models.densenet201(pretrained=False)
num_classes = 1000  # 학습 당시 사용된 클래스 수로 설정
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# 모델을 GPU로 이동
model = model.to(device)

# 가중치 불러오기 (GPU 학습된 모델을 GPU로 불러오기)
model_load_path = "/Users/byeongyeongtae/Desktop/pbl1/python_server/dog_classifier_densenet201_final_12.pth"
model.load_state_dict(torch.load(model_load_path, map_location=device))

# 평가 모드로 전환
model.eval()

print(f"모델이 {model_load_path} 경로에서 불러와졌습니다.")

# 클래스 레이블 정의 (학습 시 사용한 클래스 레이블 리스트)
class_labels = [
    "비숑", "보더콜리", "불독", "치와와", "골든 리트리버",
    "말티즈", "포메라니안", "푸들", "시바견", "시츄", "시베리안 허스키",
    "요크셔테리어"
]

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델이 기대하는 입력 크기로 변경
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet과 동일한 정규화 사용
])

# 새로운 이미지들의 경로를 폴더로 설정
new_images_dir = 'testDog'
new_images = [os.path.join(new_images_dir, img) for img in os.listdir(new_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# 이미지 분류 및 결과 출력
for img_path in new_images:
    try:
        # 이미지 파일을 열고 RGB로 변환
        image = Image.open(img_path).convert("RGB")
        # 전처리 및 배치 차원 추가
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)  # 확률 계산
            top5_prob, top5_classes = torch.topk(probabilities, 5)  # 상위 5개의 확률과 클래스 추출

        # 결과 시각화 - 예측된 클래스 인덱스를 클래스 이름으로 변환하여 출력
        plt.imshow(Image.open(img_path))
        plt.title(f"Predicted: {class_labels[top5_classes[0]]} ({top5_prob[0].item() * 100:.2f}%)\n"
                  f"2nd: {class_labels[top5_classes[1]]} ({top5_prob[1].item() * 100:.2f}%)\n"
                  f"3rd: {class_labels[top5_classes[2]]} ({top5_prob[2].item() * 100:.2f}%)\n"
                  f"4rd: {class_labels[top5_classes[3]]} ({top5_prob[3].item() * 100:.2f}%)\n"
                  f"5rd: {class_labels[top5_classes[4]]} ({top5_prob[4].item() * 100:.2f}%)\n")
        plt.axis('off')
        plt.show()

        # 메모리 정리
        torch.cuda.empty_cache()  # GPU 메모리 정리
        plt.close()  # 플롯을 닫아 메모리 확보
        gc.collect()  # 가비지 콜렉션 실행

    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
