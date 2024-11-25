# import cv2
# import mediapipe as mp
# import numpy as np
# import pandas as pd
# import time
# import os

# # MediaPipe Hands 초기화
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 최대 두 손을 감지
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # XLSX 파일 경로
# xlsx_path = '/Users/ijaein/Desktop/단어정리.xlsx'

# # 동영상 파일이 저장된 폴더 경로
# video_folder = '/Users/ijaein/Desktop/vf'

# # XLSX 파일 읽기
# df = pd.read_excel(xlsx_path)

# # 출력 디렉토리 생성
# created_time = int(time.time())
# os.makedirs('dataset_both', exist_ok=True)

# # 시퀀스 길이 고정 (예: 30 프레임)
# seq_length = 60

# # XLSX 파일의 각 행에 대해 처리 시작
# for idx, row in df.iterrows():
#     video_file = row['파일명']
#     action = row['단어명']
    
#     print(f"'{action}' 동작에 대한 좌표 추출을 시작합니다.")

#     data = []

#     video_path = os.path.join(video_folder, video_file)
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"동영상을 열 수 없습니다: {video_path}")
#         continue

#     start_time = time.time()

#     while time.time() - start_time < 4:  # 4초 동안 데이터 수집
#         ret, img = cap.read()
#         if not ret:
#             print("동영상의 끝에 도달했습니다.")
#             break

#         img = cv2.flip(img, 1)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         result = hands.process(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         if result.multi_hand_landmarks is not None:
#             joint = np.zeros((126 * 2,))  # 양손의 랜드마크 데이터 (21*3*2)
#             hand_count = 0  # 손의 개수를 추적

#             for hand_landmarks in result.multi_hand_landmarks:
#                 for j, lm in enumerate(hand_landmarks.landmark):
#                     joint[hand_count * 21 * 3 + j * 3] = lm.x  # x 좌표
#                     joint[hand_count * 21 * 3 + j * 3 + 1] = lm.y  # y 좌표
#                     joint[hand_count * 21 * 3 + j * 3 + 2] = lm.z  # z 좌표
#                 hand_count += 1

#                 # 양손의 랜드마크 그리기
#                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # 최대 두 손만 처리하도록 설정
#                 if hand_count == 2:
#                     break

#             data.append(joint)

#         cv2.imshow('img', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit()

#     # 시퀀스 길이 맞추기 (부족하면 0으로 패딩, 길면 자름)
#     while len(data) < seq_length:
#         padding = np.zeros((126 * 2,))  # 크기가 126*2인 패딩 생성
#         data.append(padding)

#     data = np.array(data[:seq_length])

#     # 고유한 파일 이름 생성 (단어명 + 타임스탬프)
#     timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
#     file_path = os.path.join('dataset_both', f'{action}_{timestamp}.npy')

#     # 새로운 데이터를 저장 (덮어쓰지 않음)
#     np.save(file_path, data)

#     cap.release()

#     print(f"'{action}' 동작에 대한 데이터 수집이 완료되었습니다.")

# # 모든 동작 수집 완료 후 종료
# cv2.destroyAllWindows()
# print("모든 동작에 대한 데이터 수집이 완료되었습니다.")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,  # 최대 두 손을 감지
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# XLSX 파일 경로
xlsx_path = '/Users/ijaein/Desktop/시현단어.xlsx'

# 동영상 파일이 저장된 폴더 경로
video_folder = '/Users/ijaein/Desktop/vf'

# XLSX 파일 읽기
df = pd.read_excel(xlsx_path)

# 출력 디렉토리 생성
created_time = int(time.time())
os.makedirs('dataset_both_시현', exist_ok=True)

# 시퀀스 길이 고정 (예: 60 프레임)
seq_length = 30

# XLSX 파일의 각 행에 대해 처리 시작
for idx, row in df.iterrows():
    video_file = row['파일명']
    action = row['단어명']
    
    print(f"'{action}' 동작에 대한 좌표 추출을 시작합니다.")

    data = []

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        continue

    while True:  # 더 이상 프레임이 없을 때까지 계속 반복
        ret, img = cap.read()
        if not ret:
            print("동영상의 끝에 도달했습니다.")
            break

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            joint = np.zeros((126 * 2,))  # 양손의 랜드마크 데이터 (21*3*2)
            hand_count = 0  # 손의 개수를 추적

            for hand_landmarks in result.multi_hand_landmarks:
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[hand_count * 21 * 3 + j * 3] = lm.x  # x 좌표
                    joint[hand_count * 21 * 3 + j * 3 + 1] = lm.y  # y 좌표
                    joint[hand_count * 21 * 3 + j * 3 + 2] = lm.z  # z 좌표
                hand_count += 1

                # 양손의 랜드마크 그리기
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 최대 두 손만 처리하도록 설정
                if hand_count == 2:
                    break

            data.append(joint)

        cv2.imshow('img', img)
        
        # 적절한 딜레이로 프레임 조절 (50ms 대기)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # 시퀀스 길이 맞추기 (부족하면 0으로 패딩, 길면 자름)
    while len(data) < seq_length:
        padding = np.zeros((126 * 2,))  # 크기가 126*2인 패딩 생성
        data.append(padding)

    data = np.array(data[:seq_length])

    # 고유한 파일 이름 생성 (단어명 + 타임스탬프)
    timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
    file_path = os.path.join('dataset_both_시현', f'{action}_{timestamp}.npy')

    # 새로운 데이터를 저장 (덮어쓰지 않음)
    np.save(file_path, data)

    cap.release()

    print(f"'{action}' 동작에 대한 데이터 수집이 완료되었습니다.")

# 모든 동작 수집 완료 후 종료
cv2.destroyAllWindows()
print("모든 동작에 대한 데이터 수집이 완료되었습니다.")