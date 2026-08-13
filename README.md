# 🍑 Peach Detection & Classification

복숭아 이미지를 대상으로 **① YOLOv8 기반 객체 탐지(Detection)** 와 **② 커스텀 CNN 기반 이미지 분류(Classification)** 를 함께 수행하는 딥러닝 파이프라인입니다.

---

## 📌 프로젝트 개요

- **탐지(Detection)**: `yolov8n.pt`(YOLOv8 nano) 모델을 이용해 이미지 내 복숭아 위치를 탐지합니다.
- **분류(Classification)**: 커스텀 CNN 모델(`train.py`, `eval.py`, `predict.py`)을 이용해 복숭아 이미지를 분류합니다. (등급/상태/품종 등 분류 기준은 프로젝트 목적에 맞게 정의)
- **데이터 증강(Augmentation)**: `main.py`에서 `albumentations` 라이브러리를 사용해 YOLO 학습용 이미지에 Crop, Flip, Brightness/Contrast 등의 증강을 적용합니다.

> ⚠️ 일부 세부 사항(분류 클래스 기준, 모델 아키텍처 등)은 코드에 명시되어 있지 않아 실제 프로젝트 내용에 맞게 보완이 필요합니다.

---

## 📁 파일 구성

| 파일 | 설명 |
|---|---|
| `yolov8n.pt` | YOLOv8 nano 사전학습/학습된 가중치 파일 |
| `yolo_setting.yaml` | YOLO 학습/검증용 데이터셋 경로 및 클래스 설정 파일 |
| `main.py` | Albumentations를 이용한 이미지 증강 파이프라인 |
| `train.py` | CNN 모델 학습(1 epoch) 함수 |
| `eval.py` | CNN 모델 검증(evaluate) 함수 |
| `predict.py` | 단일 이미지에 대한 CNN 모델 추론 함수 |
| `README.md` | 프로젝트 설명 문서 |

---

## 🗂 데이터셋 구조

`yolo_setting.yaml` 기준 데이터셋은 아래와 같은 구조를 가집니다.

```
YoloDataSet/
├── images/
│   ├── train/
│   └── valid/
└── labels/
    ├── train/
    └── valid/
```

```yaml
path: C:\Users\user\Downloads\Deep_Learning_Gicho\Data\PeachDataSet\YoloDataSet
train: images/train
val: images/valid

names:
  0: peach
```

> 클래스는 현재 `peach` 단일 클래스(0번)로 정의되어 있습니다.
> `path`는 로컬 절대경로로 지정되어 있으므로, 다른 환경에서 실행 시 본인 경로에 맞게 수정이 필요합니다.

---

## ⚙️ 요구사항

```bash
pip install torch torchvision
pip install ultralytics
pip install albumentations
pip install opencv-python matplotlib numpy
```

---

## 🚀 사용 방법

### 1. 데이터 증강 (`main.py`)

```bash
python main.py
```

`albumentations`를 사용해 RandomCrop, HorizontalFlip, RandomBrightnessContrast 등을 적용한 예시 이미지를 확인할 수 있습니다. (`Utils/Augmentation.py`에 실제 증강 파이프라인 함수들이 정의되어 있어야 합니다.)

### 2. YOLOv8 학습 / 추론

`ultralytics` 라이브러리를 이용해 아래와 같이 학습 및 추론을 진행합니다.

```python
from ultralytics import YOLO

# 학습
model = YOLO("yolov8n.pt")
model.train(data="yolo_setting.yaml", epochs=100, imgsz=640)

# 추론
results = model.predict(source="이미지_경로", save=True)
```

### 3. CNN 분류 모델 학습 / 검증 (`train.py`, `eval.py`)

`train_one_epoch`와 `evaluate` 함수를 이용해 모델을 학습/검증합니다.

```python
from train import train_one_epoch
from eval import evaluate

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
```

### 4. 단일 이미지 예측 (`predict.py`)

```python
from predict import predict_image

predict_image("테스트_이미지.jpg", model)
```

이미지를 128x128로 리사이즈하고 정규화한 뒤, 모델의 예측 클래스를 출력합니다.

---

## 🔄 딥러닝 파이프라인 순서

1. 데이터 수집
2. 데이터 전처리 (증강 포함, `main.py`)
3. 알고리즘 선택 (YOLOv8 / CNN)
4. 훈련 (`train.py` / YOLO `model.train`)
5. 검증 (`eval.py`)
6. 평가
7. 배포 / 추론 (`predict.py` / YOLO `model.predict`)

---

## 📝 참고 사항

- `predict.py`에서 사용하는 모델은 입력 크기 `128x128`, 채널 3(RGB)을 기대하는 CNN 분류 모델입니다. 실제 사용하는 모델 클래스(`model.py` 등)는 별도로 정의되어 있어야 합니다.
- `main.py`에서 import하는 `Utils.Augmentation` 모듈은 현재 업로드된 파일에 포함되어 있지 않으므로, 함께 관리해야 정상 동작합니다.
- `yolo_setting.yaml`의 `path`는 Windows 로컬 경로이므로, 다른 PC/서버에서 실행 시 수정이 필요합니다.
