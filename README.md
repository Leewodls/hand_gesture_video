## hand_gesture_video
양손 동작 데이터를 기반으로 수어를 인식하기 위해 설계된 딥러닝 기반 시스템
MediaPipe를 활용하여 손 좌표 데이터를 추출하고, TensorFlow/Keras를 사용하여 수어를 학습 및 인식하는 모델을 생성
각 손의 동작을 개별적으로 처리하고, 최종적으로 양손의 데이터를 결합하여 수어를 인식하는 모델을 제공

## 기능
- 손 좌표 데이터 추출:
  - MediaPipe를 사용하여 영상에서 양손 좌표 데이터를 추출.
  - 추출된 데이터를 .npy 형식으로 저장.
- 모델 학습:
  - 좌우 손 데이터를 개별적으로 학습하여 손 구분 모델 생성.
  - 양손 데이터를 결합하여 최종 수어 인식 모델 학습.
- 실시간 테스트:
  - OpenCV를 사용하여 실시간으로 손 동작을 추적하고 수어 인식 결과를 출력.

## 각 파일 설명
- 손구분모델생성.ipynb:
  - 좌/우 손 동작 데이터를 학습하여 손 구분 모델을 생성
- 영상추출.py:
  - 영상 데이터를 처리하여 손 좌표 데이터를 추출
- 양손합쳐추출.py:
  - 양손 데이터를 결합하여 수어 데이터 생성
- 각손테스트.py:
  - 좌/우 손 모델을 개별적으로 테스트
- 양손테스트.py:
  - 양손 데이터를 테스트하고 수어를 실시간으로 예측
 
## 기술
	•	Python: 데이터 처리 및 모델 구현
	•	MediaPipe: 손 좌표 추출
	•	TensorFlow/Keras: 딥러닝 모델 설계 및 학습
	•	OpenCV: 실시간 영상 처리


# 좌표 추출 설명
## 활용 데이터
<img width="866" alt="스크린샷 2024-11-25 오후 8 36 04" src="https://github.com/user-attachments/assets/a284a641-f099-490f-9322-e61edb6e9ac0">

## 각 손 데이터
<img width="920" alt="스크린샷 2024-11-25 오후 8 45 57" src="https://github.com/user-attachments/assets/eeb62ee0-4339-4e51-8c37-bf3e5dc0876a">

## 양손 데이터
<img width="924" alt="스크린샷 2024-11-25 오후 8 46 19" src="https://github.com/user-attachments/assets/fde9db2a-6fbb-467a-a04c-290965eff7c6">

## 데이터 전처리
<img width="931" alt="스크린샷 2024-11-25 오후 8 46 32" src="https://github.com/user-attachments/assets/f452369e-622c-4c4f-ad15-fa1c3373e736">
<img width="911" alt="스크린샷 2024-11-25 오후 8 46 54" src="https://github.com/user-attachments/assets/a418a6ab-772f-44d3-a93c-8d93c3a040c5">

## 모델 아키텍쳐 및 결과
<img width="850" alt="스크린샷 2024-11-25 오후 8 47 04" src="https://github.com/user-attachments/assets/a9d2b033-72ec-4073-b887-438953839972">
<img width="933" alt="스크린샷 2024-11-25 오후 8 47 16" src="https://github.com/user-attachments/assets/3cb5a2dc-d14a-497e-8c32-2ed237edbdd1">



