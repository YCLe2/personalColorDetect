# 웹캠을 이용한 퍼스널 컬러 분석

이 프로젝트는 OpenCV와 Dlib을 사용하여 사용자의 퍼스널 컬러 톤(예: 봄 웜, 가을 웜, 여름 쿨, 겨울 쿨)을 분석합니다. 웹캠에서 얼굴을 검출하고 피부, 눈, 눈썹 색상을 추출하여 사전 정의된 모델을 기반으로 퍼스널 컬러를 판별합니다.

---

## 주요 기능

- Dlib의 얼굴 검출기를 이용한 실시간 얼굴 탐지.
- 랜드마크 탐지를 통해 정밀한 얼굴 특징 위치 파악.
- 피부, 눈, 눈썹 색상 분석.
- 퍼스널 컬러를 아래 네 가지로 분류:
  - **봄 웜 (Spring Warm)**
  - **가을 웜 (Fall Warm)**
  - **여름 쿨 (Summer Cool)**
  - **겨울 쿨 (Winter Cool)**
- OpenCV의 GrayworldWB로 자동 화이트 밸런스 조정.

---

## 코드 설명

### **1. 라이브러리 임포트**
프로그램은 이미지 처리(`OpenCV`), 얼굴 및 랜드마크 탐지(`Dlib`), 유틸리티 함수(`iostream`, `vector`)에 필요한 라이브러리를 임포트합니다.

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/white_balance.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>


### **2.색상 추출 (getDominantColor) **
특정 이미지 영역의 주요 색상(도미넌트 컬러)을 계산하는 함수입니다.
region.convertTo(data, CV_32F); => 32비트 부동소수점으로 변환
data = data.reshape(1, data.total()); => 데이터를 1차원으로 변환