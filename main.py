# 라벨 만들기 위한 os, json
import os, json
import matplotlib.pyplot as plt
import cv2
import re
import shutil
from ultralytics import YOLO

if __name__ == "__main__":
    # 머신러닝 객체 생성 -> 훈련 -> 예측 -> 평가
    # YOLO 라이브러리 세팅(pip)
    #model = YOLO()

    yaml_path = r"./yolo_setting.yaml"
    
    # YOLO 훈련
    result = YOLO('yolov8n.pt').train(
                                        data = yaml_path,
                                        epochs = 50,
                                        imgsz = 640,
                                        batch = 16,
                                        save = True,
                                        device = 0,
                                        plots = True,
                                        name = 'peach_train01')

    print('훈련 완료')
    
    # YOLO 평가


# 딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포
