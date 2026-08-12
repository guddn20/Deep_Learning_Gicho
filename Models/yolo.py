# 머신러닝 객체 생성 -> 훈련 -> 예측 -> 평가
# YOLO 라이브러리 세팅(pip)
# model = YOLO()
from ultralytics import YOLO

def yolo_train():

    yaml_path = r"./yolo_setting.yaml"

    # YOLO 훈련
    result = YOLO("yolov8n.pt").train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=16,
        save=True,
        device=0,
        plots=True,
        name="peach_train01")

    print("훈련 완료")

