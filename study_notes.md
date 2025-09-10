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

## Day 3 : Fast R-CNN, Faster R-CNN

### Fast RCNN

- Overall Architecture
  1. 이미지를 CNN에 넣어 feature 추출 : VGG16 사용
  2. RoI Projection을 통해 feature map 상에서 RoI를 계산
  3. RoI Pooling을 통해 일정한 크기의 feature가 추출
    - 고정된 vector 얻기 위한 과정
    - SPP 사용
      - pyramid level : 1
      - Target grid size : 7 x 7 
  4. Fully connected layer 이후, Softmax Classifier과 Bounding Box Regressor
     - 클래스 개수 : C + 1개
       - 클래스 (C 개) + 배경 (1r개)

- Training

  - multi task loss 사용
    - classification loss + bounding box regression
  - Loss function
    - classification (Cross entropy)
    - bounding box regressor (Smooth L1)
  - Dataset 구성
    - IoU > 0.5: positive samples
    - 0.1 < IoU < 0.5: negative samples
    - Positive samples 25%, negative samples 75%
  - Hierarchical sampling
    - R-CNN의 경우 이미지에 존재하는 RoI를 전부 저장해 사용
    - 한 배치에 서로 다른 이미지의 RoI가 포함됨
    - Fast R-CNN의 경우 한 배치에 한 이미지의 RoI만 포함
    - 한 배치 안에서 연산과 메모리를 공유할 수 있음

### Faster RCNN

- Pipeline

  1. 이미지를 CNN에 넣어 feature maps 추출 (CNN을 한 번만 사용)
  2. RPN을 통해 RoI 계산
     - 기존의 selective search 대체
     - Anchor box 개념 사용
       - Anchor box : 각 피쳐마다 다양한 크기와 비율로 미리 정의된 박스

    - Region Proposal Network (RPN)
      2-1. CNN에서 나온 feature map을 input으로 받음. (𝐻: 세로, 𝑊: 가로, 𝐶: 채널)
      2-2. 3x3 conv 수행하여 intermediate layer 생성
      2-3. 1x1 conv 수행하여 binary classification 수행

  3. RPN에서 뽑은 N개의 박스를 대상으로 RoI Pooling을 통해 일정한 크기의 feature가 추출
  4. Fully connected layer 이후, Softmax Classifier과 Bouding Box Regressor

- NMS
  - 유사한 RPN Proposals 제거하기 위해 사용
  - Class score를 기준으로 proposals 분류
  - IoU가 0.7 이상인 proposals 영역들은 중복된 영역으로 판단한 뒤 제거

- Training

  - Region Proposal Network (RPN)
    - RPN 단계에서 classification과 regressor학습을 위해 앵커박스를 positive/- negative samples 구분
  - 데이터셋 구성
    - IoU > 0.7 or highest IoU with GT: positive samples
    - IoU < 0.3: negative samples
    - Otherwise : 학습데이터로 사용 X
  - RPN 단계에서 classification과 regressor학습을 위해 앵커박스를  positive/- negative samples 구분

  - Region proposal 이후

    - Fast RCNN 학습을 위해 positive/negative samples로 구분
    - 데이터셋 구성
      - IoU > 0.5: positive samples → 32개
      - IoU < 0.5: negative samples → 96개
      - 128개의 samples로 mini-bath 구성
    - Loss 함수
      - Fast RCNN과 동일

## Day 4 : Object Detection

### Object Detection을 위한 라이브러리
 

| | MMDetection | Detectron2 |
| - | - |- |
| **특징** | - 전체 프레임워크를 모듈 단위로 분리해 관리할 수 있음  <br> - 많은 프레임워크를 지원함 <br> - 다른 라이브러리에 비해 빠름 | - 전체 프레임워크를 모듈 단위로 분리해 관리할 수 있음 <br> - OD 외에도 Segmentation, Pose prediction 등의 알고리즘을 지원함 |
| **지원 모델** | - Fast R-CNN <br> - SSD <br> - YOLO v3 <br> - DETR <br> | - Faster R-CNN <br> - RetinaNet <br> - Mask R-CNN <br> - DETR <br> |

### MMDetection

https://github.com/open-mmlab/mmdetection

- Pytorch 기반의 Object Detection 오픈소스 라이브러리
- 커스텀이나 튜닝을 위해서는 라이브러리를 완벽히 이해하는게 많이 요구된다

- 하나의 config 파일에서 Backbone, Neck, Densehead, RoIHead 모든 파이프라인을 커스터마이징 가능
  - Backbone : 입력 이미지를 특정 map으로 변형
  - Neck : backbone과 head를 연결, Feature map을 재구성 (ex. FPN)
  - DenseHead : Feature map의 dense location을 수행하는 부분
  - RoIHead : RoI 특징을 입력으로 받아 box 분류, 좌표 회귀 등을 예측하는 부분

- Pipeline
  - 라이브러리, 모듈 import
  - config 파일 불러오기
  - config 파일 수정
  - 모델, 데이터셋 build
  - 학습

- Config file 
  - 구조
    - configs를 통해 데이터셋부터 모델, scheduler, optimizer 정의 가능
    - 특히, configs에는 다양한 object detection 모델들의 config 파일들이 정의돼 있음
    - 그 중, configs/base/ 폴더에 가장 기본이 되는 config 파일이 존재
  - dataset, model, schedule, default_runtime 4가지 기본 구성요소 존재
  - 각각의 base/ 폴더에는 여러 버전의 config들이 담겨있음
    - Dataset – COCO, VOC, Cityscape 등
    - Model – faster_rcnn, retinanet, rpn 등
  - 틀이 갖춰진 config를 상속 받고, 필요한 부분만 수정해 사용함

- Dataset
  - samples_per_gpu
  - workers_per_gpu
  - train
  - val
  - test

  - train  pipeline
    - Load Image From File
    - Load Annotations
    - Resize
    - RandomFlip
    - Normalize
    - Pad
    - DefaultFormat Bundle
    - Collect

- Model

  - 2 stage model
    - type
      - 모델 유형
      - ex) FasterRCNN, RetinaNet...
    - backbone
      - 인풋 이미지를 feature map으로 변형해주는 네트워크
      - ex) ResNet, ResNext, HRNet...
    - neck
      - Backbone과 head를 연결
      - Feature map을 재구성
      - ex) FPN, NAS_FPN, PAFPN...
    - rpn_head
      - Region Proposal Network
      - RPNHead, Anchor_Free_Head...
      - Anchor_generator
      - Bbox_coder
      - Loss_cls
      - Loss_bbox
    - roi_head
      - Region of Interest
      - StandardRoIHead, CascadeRoIHead...
      - bbox_roi_extractor
      - bbox_head
    - bbox_head
    - train_cfg
    - test_cfg

  - 커스텀 backbone 모델 등록
    1. 새로운 backbone 등록
    2. 모듈 import
    3. 등록한 backbone 사용

- Runtim settings
  - Optimizer
    - ex) SGD, Adam...
  - Training schedules
    - learning rate
    - runner

## Day 5 : Detectron2

- Facebook AI Research의 Pytorch 기반 라이브러리
- Object Detection 외에도 Segmentation, Pose prediction 등 알고리즘도 제공

- pipeline
  - Setup Config
  - Setup Trainer
    - build_model
    - build_detection_train / test_loader
    - build_optimizer
    - build_Ir_scheduler
  - Start Training

- Config File
  - MMDetection과 유사하게 config 파일을 수정, 이를 바탕으로 파이프라인을 build하고 학습함
  - 틀이 갖춰진 기본 config를 상속 받고, 필요한 부분만 수정해 사용함

- Dataset
  - config
    - 데이터셋, 데이터로더와 관련된 config
    - TRAIN, TEST에 각각 등록한 train 데이터셋과 test 데이터셋의 이름을 입력함
    - 데이터셋, 데이터로더와 관련된 config
    - TRAIN, TEST에 각각 등록한 train 데이터셋과 test 데이터셋의 이름을 입력함

- Dataset 등록
  - 커스텀 데이터셋을 사용하고자 할 때는 데이터셋을 등록해야함
  - (옵션) 전체 데이터셋이 공유하는 정보 (ex. class명, 파일 디렉토리 등)을 메타 데이터로 등록할 수 있음

- Model
  - Backbone
    - 인풋 이미지를 특징맵으로 변형해주는 네트워크
    - ex) ResNet, RegNet 
  - FPN
    - Backbone과 head를 연결, Feature map을 재구성
  - ANCHOR_GENERATOR
  - RPN
  - ROI_HEADS
  - ROI_BOX_HEAD

- Solver
  - LR_SCHEDULER
  - WEIGHT_DECAY
  - CLIP_GRADIENTS
