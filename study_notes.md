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

## Day 2 : 2 Stage Detectors, R-CNN, SPPNet

### 접근 전략

- 사람이 객체를 검출하는 방법
  - 객체가 어디에 있는지, 무엇이 배경과 다른지, 객체가 있을만한 위치를 생각
  - 객체에 대해서 생각

- 인공지능이 객체를 검출하는 방법
  - 입력 이미지가 주어졌을 때, 어떠한 계산을 통해서 객첼르 검출
  - 객체가 있을 법한 위치를 예측
  - 객체가 어떤 객체인지를 식별

### R-CNN

- R-CNN 대략적인 과정
  1. 입력 이미지를 받기
  2. 후보 영역을 추출
    - 후보 영역을 추출하는 방법
      - 슬라이딩 윈도우
      - Selective Search
        - 이미지를 무수히 많은 작은 영역으로 나눈다
        - 이 영역을 점차 통합하는 방식
  3. 후보 영역의 크기를 조절해 모두 동일한 사이즈로 변형: 각기 다른 사이즈의 후보영역을 동일한 사이즈로 변환 (warping)
  4. 변환된 후보 영역을 CNN에 넣어 feature 추출
  5. 추출된 feature를 이용해
    - SVM으로 객체 분류
    - Bounding box regression으로 위치 조정

- Pipeline
  1. 입력 이미지를 받기
  2. Selective Search를 통해 약 2000개의 RoI(Region of Interest)를 추출
  3. RoI(Region of Interest)의 크기를 조절해 모두 동일한 사이즈로 변형: 각기 다른 사이즈의 후보영역을 동일한 사이즈로 warping 한다. (CNN의 마지막인 FC layer의 입력 사이즈가 고정이므로 Wraping 해야함)
  4. RoI를 CNN에 넣어, feature를 추출
    - 각 regoin 마다 4096 dim feature vector 추출 (2000 x 4096)
    - Pretrained AlexNet 구조 활용
      - AlexNet 마지막에 FC layer 추가
      - 필요에 따라 Finetuning 진행
  5. CNN을 통해 나온 feature를 
    1. SVM에 넣어 분류
       - input
         - 2000 x 4096 features
       - output
         - Class (C+1) + Confidence scores
         - 클래스 개수 (C개) + 배경 여부 (1개)      
    2. regression을 통해 bounding box를 예측

- 단점 (초기 개념이라 많음)
  - 2000r개의 region을 각각 CNN 통과
  - 강제 Warping, 성능 하락 가능
  - CNN, SVM classifier, bounding box regressor, 따로 학습
  - End to End 가 아님

### SPPNet

- R-CNN의 한계점
  - Convolution Network의 입력 이미지가 고정 -> 이미지를 고정된 크기로 자르거나(crop) 비율을 조정(warp)해야함
  - RoI(Region of Interest)마다 CNN통과 -> 하나의 이미지에 대해서 2000번 CNN을 통과해야함

- Pipeline
  - 한 번의 Conv 연산
  - 이미지에서 2000개의 ROI 생성
  - 각 ROI에 대해 Spatial Pyramid Pooling (SPP) 적용 → 고정 크기의 feature vector 생성
  - 고정 크기 feature를 fully connected layer에 전달하여 객체 분류 및 bounding box regression 수행

- Spatial Pyramid Pooling
  - Conv Layer들을 거쳐서 추출된 피쳐맵을 다양한 타겟 사이즈로 비닝 진행
  - 비닝된 셀마다 max pooking, avg pooling 진행