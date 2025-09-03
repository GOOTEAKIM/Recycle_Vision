# 재활용 쓰레기를 활용한 딥러닝 - Detection

## Day 1 : 개요, 소개

### Computer vision task 종류

- Classifcation : 사진이 주어졌을 때 무엇인지 예측하는 task
- Object Detection : 이미지 속세어 객체를 식별하는 task, 객체가 어디에 있고, 그 객체가 무엇인지 식별
  - ex)
    - 테슬라의 자율주행 자동차
    - OCR
    - x ray 사진에서 병의 위치를 찾는 task
    - CCTV에서 사람을 검출하는 task
- Segmantic Segmentation : 객체의 영역을 구분하는 task, 같은 클래스를 갖는 객체끼리는 구분이 없다
- Instance Segmentation : 객체의 영역을 구분하는 task, 같은 클래스의 객체도 구분한다

### Object Detection의 역사

- R-CNN
- Fast R-CNN
- Faster R-CNN
- YOLO v1
- SSD
- YOLO v2
- FPN
- RetinaNet
- YOLO v3
- PANet
- EfficientDet
- Swin-T

### Object Detection의 평가 지표

- mAP (mean average precision) : **각 클래스당 AP의 평균**
- mAP를 계산하기 위해 필요한 개념
  - Confusion matrix
    - 예측 결과의 case
      - TP (True Positive) : 검출 되어야 할 것이 검출
      - FP (False Positive) : 검출 되지 않아야 할 것이 검출
      - FN (False Negative) : 검출 되어야 할 것이 검출되지 않음
      - TN (True Negative) : 검출되지 말아야 할 것이 검출
  - Precision : 모델의 **예측 관점**에서 정의한 matric, 모델이 positive라고 예측한 모든 케이스 중 옳게 예측한 경우
  - Recall : 정답 관점의 matric, == TP / (TP + FN)
  - PR curve : Recall 값에 따른 Precision 값의 변화의 그래프
  - IOU : Ground Truth 박스와 Predict 박스 두 개의 전체 영역에 겹치는 영역
  - FPS : 초당 처리 가능한 프레임 숫자, 크면 클수록 빠른 모델
  - FLOPs : 모델의 연산량을 측정할 수 있는 평가지표, 연산량 횟수 (곱하기, 덧셈, 빼기 등)
    - ex) MUL (3 x 2, 2 x 3) == 3 x 3
      - 곱셈 : 3 x 3 x 2 = 18
      - 덧셈 : 3 x 3 x 1 = 9
      - Flops == 18 + 9 = 27 

### Object Detection Library

- MMDetection: OpenMMLab에서 진행하는 Object Detection Libary
- Detectron2: Detectron2는 페이스북 AI 리서치의 라이브러리로 pytorch 기반의 Object Detection Libary
- YOLOv5: COCO 데이터셋으로 사전 학습된 모델로 수천 시간의 연구와 개발에 걸쳐 발전된 Object Detection 모델