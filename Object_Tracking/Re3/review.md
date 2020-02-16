# Re3: Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects

![](output.gif)

- Author
  - Daniel Gordon, Ali Farhadi, Dieter Fox
- Title of Conference
  - IEEE Robotics and Automation Letters (RA-L)
- Materials
  - [Code](https://gitlab.com/danielgordon10/re3-tensorflow)
  - [YouTube](https://www.youtube.com/watch?v=RByCiOLlxug)

## Abstract

- Robust object tracking을 위해서는 대상체의 모양, 움직임, 시간에 따른 변화를 이해하는 것이 필수적.
  - 따라서 tracer는 새로운 관찰에 맞춰 자신의 모델을 수정할 수 있어야 한다.
  - 이것이 가능한 실시간 deep object tracker인 Re3를 제안.
- 특정 object들만 트래킹하는 것이 아니라 generic tracker를 학습시키고, 효율적으로 tracker를 on-the-fly로 업데이트.
  - 한 번의 forward pass만으로 tracking과 appreance model을 업데이트한다.
  - 높은 성능과 빠른 속도(150fps) 달성.
- 또한 일시적인 occlusion을 다른 tracker들에 비해 잘 다룬다. 

## 1. Introduction

- 주로 로보틱스 분야에서는 알려진 object들이나 특정한 object 객체들을 추적하는 tracker들을 개발해 왔다.
  - 이런 세팅은 tracker들은 offline으로 디자인되거나 학습될 수 있다는 이점이 있고, object들의 shape model이 보통 사용가능할 때도 좋다. 
  - 그러나 많은 시나리오들에서 어떤 object들이 트래킹될지를 미리 특정하는 것은 적절치 않다. 
- 여기서는 RGB 비디오 데이터를 가지고 포괄적인 object tracking을 하는 데 집중한다. 
- 이 논문에서, 우리는 스티리밍 데이터에서 작동하는 tracker만 고려한다.
  - 즉, 주어진 현재 또는 미래의 관찰을 가지고 과거의 추측을 수정할 수 없다. 
- 지금의 포괄적인 2D 이미지 트래킹 시스템들은 대부분 tracker를 온라인으로 학습시키는 데 의존한다. 
- 인기 있는 트래킹 알고리즘 패러다임은 detection을 통한 트래킹이다.
  - Object-specific한 detector를 학습시키고, 각 프레임에서 object의 새로운 모습을 업데이트 하는 식.
  - 이 기법의 단점은 tracker를 업데이트 하는 데 보통 많은 연산이 필요하다는 점이다. 
  - 반대로, 어떤 object tracker들은 dectector를 offline으로 학습시키는 대신 적은 object type들에 대해서만 작동한다. 
- 이러한 단점들을 극복한 빠르고 정확한 generic object tracker, Re3을 제안한다.
  - Re3은 *Re*al-time, *Re*current, *Re*gression-based 이란 뜻.

## 2. Related Work

#### Online-trained trackers

#### Offline-trained trackers

#### Hybrid trackers

## 3. Method

### A. Object Appearance Embedding

#### Network Inputs

#### Skip Connections

### B. Recurrent Specifications

#### Recurrent Structure

#### Network Outputs

#### Unrolling During Training

#### Learning to Fix Mistakes

### C. Training Procedure

#### Training from Video Sequences

#### Training from Synthetic Sequences

#### Tracking at Test Time

#### Implementation Details