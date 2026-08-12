# 라벨 만들기 위한 os, json
import os, json
import matplotlib.pyplot as plt
import cv2
import re
import shutil
from ultralytics import YOLO
from Models.yolo import yolo_train

if __name__ == "__main__":

    #yolo_train()
    
    # YOLO 평가
    source = r"./ppch.jpg"  # 복숭아 이미지

    model = YOLO("./runs/detect/peach_train01-6/weights/best.pt")

    model.predict(source=source,
                  device=0,
                  save=True)

    # 성능 개선 방법
    # 1. 데이터 수집
    # 2. augmentation
    # 3. 에포크 ↑

# 딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포
