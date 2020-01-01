# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

- Author
  - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
  - Microsoft Research
- Title of Conference(Journal)
  - NIPS 2015

## Abstract

- Region proposal이 최신 detection network들의 병목.
- Region Proposal Network (RPN)이라는 네트워크를 제안함. 
  - 각각의 위치에 대해 object bound와 objectness score를 동시에 예측하는 네트워크.
- RPN이 제안한 위치에 집중해서 Fast R-CNN이 detection.
- 기존 방법들에 비해 빠르고 정확하며 proposal의 수도 훨씬 적게 나온다. 

## 1. Introduction

- 최신 object detection method들은 region proposal에서 심각한 병목이 있는 상황. 
- Object detection 네트워크와 convolutional layer들을 공유하는 RPN이라는 새로운 방법을 제안함. 
- Fast R-CNN과 같은 영역 기반 detector들에서 사용하는 conv feature map들을 region proposal들을 생성하는 데도 사용할 수 있다는 것이 핵심. 
- RPN은 conv feature들 위에 있는 두 개의 conv layer들로 이루어짐.
  - 하나는 conv map의 각 위치를 짧은(e.g., 256-d) feature vector로 인코딩.
  - 다른 하나는 conv map의 각 위치마다, ![](https://latex.codecogs.com/svg.latex?k)개의 region proposal에 대한 objectness score와 regress된 bound를 출력.
- RPN은 FCN의 일종으로 볼 수 있음.
- Training 단계에서는 region proposal에 대한 fine-tuning과 object detection에 대한 것을 번갈아 가면서 한다. 

## 2. Related Work

- Skip.

## 3. Region Proposal Networks

- Region Proposal Network (RPN)은 임의 크기의 이미지를 입력으로 갖고 직사각형의 object proposal들과 이들의 objectness score들을 출력으로 갖는다.
- 이 과정을 fully-convolutional network로.

![Figure1](/Users/starlett/codes/paper_review/Faster_R-CNN-_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks/Figure1.png)

- Region proposal 생성을 위해, 가장 마지막으로 공유되는 conv layer에서 나온 feature map 위에 작은 네트워크를 둔다.
- 이 네트워크는 입력으로 들어온 feature map의 ![](https://latex.codecogs.com/svg.latex?n\times{n}) spatial window 에 대해 fully connected. (논문에서 ![](https://latex.codecogs.com/svg.latex?n=3))
- 각 sliding window는 lower-dimensional vector에 대응.
- 이 vector가 box-regression layer (reg)와 box-classification layer (cls)라는 두 개의 fully-connected layer의 입력이 된다.
- 구현은 ![](https://latex.codecogs.com/svg.latex?n\times{n}) conv layer (+ReLU) 하나 뒤에 ![](https://latex.codecogs.com/svg.latex?1\times{1}) conv layer 두 개가 오는 식으로. 

### Translation-Invariant Anchors

- 각 sliding-window 위치에서 ![](https://latex.codecogs.com/svg.latex?k)개의 region proposal이 예측된다.
  - 따라서 reg layer에서는 ![](https://latex.codecogs.com/svg.latex?k)개의 박스들의 좌표를 나타내는 ![](https://latex.codecogs.com/svg.latex?4k)개의 값이 나오고, cls layer에서는 object일 확률과 그렇지 않을 확률을 나타내는 ![](https://latex.codecogs.com/svg.latex?2k)개의 값이 나온다. 
- Anchor라고 불리는 ![](https://latex.codecogs.com/svg.latex?k)개의 reference box들을 기준으로 proposal이 생성된다. 
  - 각 anchor는 sliding window의 중심에 위치해 있으며, 크기와 가로세로 비율이 서로 다르다. 
  - 이런 방식은 translation invariant해서, 그렇지 않을 때에 비해 훨씬 효율적이다.

### A Loss Function for Learning Region Proposals

- RPN의 학습을 위해 각각의 anchor에 대해 object인지 아닌지를 나타내는 binary class label을 단다.

- Positive label이 붙는 상황은 다음 두 가지 경우 중 하나.

  - Ground-truth box와의 Intersection-over-Union (IOU)가 가장 높은 anchor(들).
  - 어떤 ground-truth box와의 IOU가 0.7을 넘는 anchor.

- Negative label은 모든 ground-truth box와의 IOU가 0.3 미만인 anchor.

- 나머지는 training 과정에서 사용하지 않음.

- 하나의 이미지에 대한 loss는 다음과 같이 정의됨.

  ![](https://latex.codecogs.com/svg.latex?L(\{p_i\},\{t_i\})=\frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_ip_i^*L_{reg}(t_i,t_i^*))

  - ![](https://latex.codecogs.com/svg.latex?i)는 mini-batch 내에서 anchor의 index
  - ![](https://latex.codecogs.com/svg.latex?p_i)는 RPN의 objectness 예측값, ![](https://latex.codecogs.com/svg.latex?p_i^*)는 ground-truth.
  - ![](https://latex.codecogs.com/svg.latex?t_i)는 예측된 bounding box의 좌표값을 나타내는 4D vector, ![](https://latex.codecogs.com/svg.latex?t_i^*)는 ground-truth. (positive anchor일 때만)
  - ![](https://latex.codecogs.com/svg.latex?L_{cls})는 object인지 아닌지에 대한 log loss.
  - ![](https://latex.codecogs.com/svg.latex?L_{reg}(t_i,t_i^*)=R(t_i-t_i^*)). ![](https://latex.codecogs.com/svg.latex?R)은 Fast R-CNN에서 사용된 smooth L1 함수다.

- Regression을 위해 네 좌표를 anchor에 대해 매개변수화한다. 

![](formula1.png)

- ![](https://latex.codecogs.com/svg.latex?k)개의 regressor가 weight값을 모두 따로 가진다.

### Optimization

- SGD를 이용해 학습.
- 각 mini-batch는 한 이미지에서 나오는 여러 개의 positive/negative anchor들로 이루어진다.
- 모든 anchor에 대해서 학습하면 negative anchor들이 훨씬 더 많아서 bias가 생기기 때문에, 한 이미지에 있는 256개의 anchor들을 랜덤으로 샘플링해서 mini-batch를 구성하는데, 이 때 positive와 negative anchor들의 비율을 1:1로 맞춘다.
  - 만약 positive sample의 수가 128개가 안 되면 빈 자리는 negative sample로 채운다.
- 학습 전에 모든 새로운 layer들의 weight을 zero-mean Gaussian distribution으로 초기화한다. std=0.01.
- 다른 layer들, 즉 공유 layer들은 ImageNet classification을 위해 pre-train된 애를 사용. 
- 60k개의 mini-batch들에 대해 lr = 0.001
- 20k개의 mini-batch들에 대해 lr = 0.0001
- momentum = 0.9, weight decay = 0.0005

### Sharing Convolutional Features for Region Proposal and Object Detection

- Detection network는 Fast R-CNN의 것을 사용했다. 

- RPN과 Fast R-CNN은 따로 학습되기 때문에 conv layer들을 서로 다른 방식으로 수정한다.

- Fast R-CNN의 학습은 고정된 object proposal에 의존하기 때문에, Fast R-CNN을 학습시키면서 proposal 매커니즘을 변경하면 제대로 converge하지 않을 가능성이 높다.

- 이 optimization은 future work로 미루고, 여기서는 alternating optimization을 통해 shared feature들을 최대한 잘 학습시키는 알고리즘을 제안한다.

  1. 먼저 RPN을 앞서 서술한 대로 학습시킨다. 이 네트워크는 ImageNet으로 pre-train된 모델로 초기화되어 있고, region proposal을 학습하면서 전체적으로 fine-tune된다. 

  2. 1단계 RPN이 생성한 proposal들을 바탕으로, 분리된 detection network를 Fast R-CNN으로 학습시킨다. 이 때도 역시 ImageNet으로 pre-train된 모델로 초기화시킨다. 이 단계까지는 conv layer들을 공유하지 않는다.
  3. Detector network를 RPN 초기화에 사용한다. 이 때, conv layer들을 고정시켜 놓고 RPN에 대해서만 fine-tune한다. 
  4. Conv layer들을 고정된 채로 두고, Fast R-CNN의 fc layer들만 fine-tune한다.

### Implementation Details



## 4. Experiments

## 5. Conclusion