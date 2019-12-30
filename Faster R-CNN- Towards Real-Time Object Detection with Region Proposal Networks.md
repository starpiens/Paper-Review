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
  - 다른 하나는 conv map의 각 위치마다, k개의 region proposal에 대한 objectness score와 regress된 bound를 출력.
- RPN은 FCN의 일종으로 볼 수 있음.
- Training 단계에서는 region proposal에 대한 fine-tuning과 object detection에 대한 것을 번갈아 가면서 한다. 

## 2. Related Work

## 3. Region Proposal Networks

## 4. Experiments

## 5. Conclusion