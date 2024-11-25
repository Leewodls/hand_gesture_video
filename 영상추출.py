# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import pandas as pd
# # import time
# # import os

# # # MediaPipe Hands 초기화
# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(
# #     max_num_hands=2,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5
# # )

# # # XLSX 파일 경로
# # xlsx_path = '/Users/ijaein/Desktop/단어정리.xlsx'

# # # 동영상 파일이 저장된 폴더 경로
# # video_folder = '/Users/ijaein/Desktop/vf'

# # # XLSX 파일 읽기
# # df = pd.read_excel(xlsx_path)

# # # 출력 디렉토리 생성
# # created_time = int(time.time())
# # os.makedirs('dataset_v', exist_ok=True)

# # # XLSX 파일의 각 행에 대해 처리 시작
# # for idx, row in df.iterrows():
# #     video_file = row['파일명']
# #     action = row['단어명']
    
# #     print(f"'{action}' 동작에 대한 좌표 추출을 시작합니다.")

# #     all_data_right_hand = []
# #     all_data_left_hand = []

# #     video_path = os.path.join(video_folder, video_file)
# #     cap = cv2.VideoCapture(video_path)

# #     if not cap.isOpened():
# #         print(f"동영상을 열 수 없습니다: {video_path}")
# #         continue

# #     data_right_hand = []
# #     data_left_hand = []

# #     start_time = time.time()

# #     while time.time() - start_time < 4:  # 4초 동안 데이터 수집
# #         ret, img = cap.read()
# #         if not ret:
# #             print("동영상의 끝에 도달했습니다.")
# #             break

# #         img = cv2.flip(img, 1)
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         result = hands.process(img)
# #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# #         if result.multi_hand_landmarks is not None:
# #             for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
# #                 hand_label = hand_info.classification[0].label
# #                 joint = np.zeros((21, 4))
# #                 for j, lm in enumerate(hand_landmarks.landmark):
# #                     joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

# #                 v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
# #                 v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
# #                 v = v2 - v1
# #                 v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

# #                 # 내적 값 클리핑
# #                 dot_product = np.einsum('nt,nt->n', v, v)
# #                 dot_product = np.clip(dot_product, -1.0, 1.0)

# #                 angle = np.arccos(dot_product)
# #                 angle = np.degrees(angle)

# #                 d = np.concatenate([joint.flatten(), angle])

# #                 if hand_label == 'Right':
# #                     data_right_hand.append(d)
# #                 else:
# #                     data_left_hand.append(d)

# #             # 양손의 랜드마크 그리기
# #             for hand_landmarks in result.multi_hand_landmarks:
# #                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #         cv2.imshow('img', img)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             cap.release()
# #             cv2.destroyAllWindows()
# #             exit()

# #     # 고유한 파일 이름 생성 (단어명 + 타임스탬프)
# #     timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
# #     file_path_right = os.path.join('dataset_v', f'{action}_{timestamp}_right.npy')
# #     file_path_left = os.path.join('dataset_v', f'{action}_{timestamp}_left.npy')

# #     # 새로운 데이터를 저장 (덮어쓰지 않음)
# #     if len(data_right_hand) > 0:
# #         np.save(file_path_right, np.array(data_right_hand))
    
# #     if len(data_left_hand) > 0:
# #         np.save(file_path_left, np.array(data_left_hand))

# #     cap.release()

# #     print(f"'{action}' 동작에 대한 데이터 수집이 완료되었습니다.")

# # # 모든 동작 수집 완료 후 종료
# # cv2.destroyAllWindows()
# # print("모든 동작에 대한 데이터 수집이 완료되었습니다.")

# #<고정된 시퀀스 길이로 수정 / 데이터 없는 부분 0으로 채우기>
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
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # XLSX 파일 경로
# xlsx_path = '/Users/ijaein/Desktop/단어정리.xlsx'

# # 동영상 파일이 저장된 폴더 경로
# video_folder = '/Users/ijaein/Desktop/vf'

# # XLSX 파일 읽기
# df = pd.read_excel(xlsx_path)

# # 출력 디렉토리 생성
# created_time = int(time.time())
# os.makedirs('dataset_v2', exist_ok=True)

# # 시퀀스 길이 고정 (예: 30 프레임)
# seq_length = 60

# # XLSX 파일의 각 행에 대해 처리 시작
# for idx, row in df.iterrows():
#     video_file = row['파일명']
#     action = row['단어명']
    
#     print(f"'{action}' 동작에 대한 좌표 추출을 시작합니다.")

#     all_data_right_hand = []
#     all_data_left_hand = []

#     video_path = os.path.join(video_folder, video_file)
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"동영상을 열 수 없습니다: {video_path}")
#         continue

#     data_right_hand = []
#     data_left_hand = []

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
#             for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
#                 hand_label = hand_info.classification[0].label
#                 joint = np.zeros((21, 4))
#                 for j, lm in enumerate(hand_landmarks.landmark):
#                     joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

#                 # 각 손의 랜드마크 데이터를 21개씩 저장
#                 if hand_label == 'Right':
#                     data_right_hand.append(joint.flatten())
#                 else:
#                     data_left_hand.append(joint.flatten())

#             # 양손의 랜드마크 그리기
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         cv2.imshow('img', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             exit()

#     # 시퀀스 길이 맞추기 (부족하면 0으로 패딩, 길면 자름)
#     while len(data_right_hand) < seq_length:
#         data_right_hand.append(np.zeros(84))  # 21개 랜드마크 * 4 (x, y, z, visibility)
#     while len(data_left_hand) < seq_length:
#         data_left_hand.append(np.zeros(84))

#     data_right_hand = np.array(data_right_hand[:seq_length])
#     data_left_hand = np.array(data_left_hand[:seq_length])

#     # 고유한 파일 이름 생성 (단어명 + 타임스탬프)
#     timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
#     file_path_right = os.path.join('dataset_v2', f'{action}_{timestamp}_right.npy')
#     file_path_left = os.path.join('dataset_v2', f'{action}_{timestamp}_left.npy')

#     # 새로운 데이터를 저장 (덮어쓰지 않음)
#     if len(data_right_hand) > 0:
#         np.save(file_path_right, data_right_hand)
    
#     if len(data_left_hand) > 0:
#         np.save(file_path_left, data_left_hand)

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
    max_num_hands=2,
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
os.makedirs('dataset_시현', exist_ok=True)

# 시퀀스 길이 고정 (예: 60 프레임)
seq_length = 60

# 배속 설정 (0.5배속)
speed_multiplier = 0.5

# FPS 설정 (기본적으로 30fps로 가정)
fps = 30
frame_delay = int((1 / (fps * speed_multiplier)) * 1000)  # ms 단위 딜레이

# XLSX 파일의 각 행에 대해 처리 시작
for idx, row in df.iterrows():
    video_file = row['파일명']
    action = row['단어명']
    
    print(f"'{action}' 동작에 대한 좌표 추출을 시작합니다.")

    all_data_right_hand = []
    all_data_left_hand = []

    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        continue

    data_right_hand = []
    data_left_hand = []

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
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = hand_info.classification[0].label
                joint = np.zeros((21, 4))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # 각 손의 랜드마크 데이터를 21개씩 저장
                if hand_label == 'Right':
                    data_right_hand.append(joint.flatten())
                else:
                    data_left_hand.append(joint.flatten())

            # 양손의 랜드마크 그리기
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('img', img)

        # 0.5배속을 적용하기 위해 딜레이를 추가 (프레임 딜레이 계산)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # 시퀀스 길이 맞추기 (부족하면 0으로 패딩, 길면 자름)
    while len(data_right_hand) < seq_length:
        data_right_hand.append(np.zeros(84))  # 21개 랜드마크 * 4 (x, y, z, visibility)
    while len(data_left_hand) < seq_length:
        data_left_hand.append(np.zeros(84))

    data_right_hand = np.array(data_right_hand[:seq_length])
    data_left_hand = np.array(data_left_hand[:seq_length])

    # 고유한 파일 이름 생성 (단어명 + 타임스탬프)
    timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
    file_path_right = os.path.join('dataset_시현', f'{action}_{timestamp}_right.npy')
    file_path_left = os.path.join('dataset_시현', f'{action}_{timestamp}_left.npy')

    # 새로운 데이터를 저장 (덮어쓰지 않음)
    if len(data_right_hand) > 0:
        np.save(file_path_right, data_right_hand)
    
    if len(data_left_hand) > 0:
        np.save(file_path_left, data_left_hand)

    cap.release()

    print(f"'{action}' 동작에 대한 데이터 수집이 완료되었습니다.")

# 모든 동작 수집 완료 후 종료
cv2.destroyAllWindows()
print("모든 동작에 대한 데이터 수집이 완료되었습니다.")