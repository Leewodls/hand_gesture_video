import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from PIL import ImageFont, ImageDraw, Image

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 학습된 모델 로드 (오른손 모델과 왼손 모델 각각 로드)
best_right_model = tf.keras.models.load_model('right_model_05.keras')
best_left_model = tf.keras.models.load_model('left_model_05.keras')

# 동작 리스트 정의 (모델 학습 시 사용한 클래스와 동일하게)
actions_right = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '112', '119', '1000', '10000', '가슴', '가시', '강', '걸렸다', '경찰', '곰', '공원', '놀이터', '구청', '귀', '금가다', '급류', '깔리다', '납치', '낫', '낯선남자', '낯선여자', '내일', '냄새나다', '누나', '누수', '다음', '달(월)', '독극물', '동생', '동전', '뒤통수', '등', '딸', '뜨거운물', '말벌', '머리', '목', '무릎', '물', '배', '복부', '벌', '범람', '복통', '볼', '삼키다', '수요일', '숨을안쉬다', '신고하세요(경찰)', '아들', '아빠', '알려주세요', '앞', '약국', '어깨', '어제', '어지러움', '언니', '얼굴', '엄마', '연기', '열', '옆쪽', '오른쪽', '오른쪽-귀', '오른쪽-눈', '오빠', '월요일', '위', '이마', '이물질', '주', '지난', '창백하다', '총', '친구', '코', '토하다', '할머니', '할아버지', '허리', '허벅지', '형', '호흡곤란', '호흡기', '화요일', '화장실']
actions_left = ['왼쪽-귀', '왼쪽-눈']

# 시퀀스 길이 고정 (모델 학습 시 사용한 값과 동일하게)
seq_length = 60
expected_feature_dim = 84  # 각 손 모델에서 기대하는 입력 데이터의 feature dimension (21*3)

# 한글 폰트 경로 설정
font_path = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
font = ImageFont.truetype(font_path, 30)

# 웹캠 열기
cap = cv2.VideoCapture(1)


# 실시간 데이터를 저장할 리스트 (오른손과 왼손 각각)
seq_data_right = []
seq_data_left = []

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # 손의 랜드마크 추출
    if result.multi_hand_landmarks is not None:
        joint_right = None
        joint_left = None

        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            joint = np.zeros((21 * 3,))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j * 3] = lm.x
                joint[j * 3 + 1] = lm.y
                joint[j * 3 + 2] = lm.z

            # 손의 종류에 따라 데이터를 나눔
            if hand_handedness.classification[0].label == "Right":
                joint_right = joint
            elif hand_handedness.classification[0].label == "Left":
                joint_left = joint

            # 손의 랜드마크 그리기
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 오른손 데이터 추가
        if joint_right is not None:
            seq_data_right.append(joint_right)
            if len(seq_data_right) > seq_length:
                seq_data_right.pop(0)

        # 왼손 데이터 추가
        if joint_left is not None:
            seq_data_left.append(joint_left)
            if len(seq_data_left) > seq_length:
                seq_data_left.pop(0)

        # 오른손 예측
        if len(seq_data_right) == seq_length:
            input_data_right = np.expand_dims(np.array(seq_data_right), axis=0)

            # feature_dim이 모델 기대 크기와 다르면 0으로 패딩
            current_feature_dim = input_data_right.shape[2]
            padding_dim = expected_feature_dim - current_feature_dim
            if padding_dim > 0:
                input_data_right = np.pad(input_data_right, ((0, 0), (0, 0), (0, padding_dim)), 'constant')

            # 예측 수행
            y_pred_right = best_right_model.predict(input_data_right).squeeze()

            # 예측 결과에 따라 동작 결정
            action_idx = np.argmax(y_pred_right)
            action_right = actions_right[action_idx]
            confidence_right = y_pred_right[action_idx]

            # 화면에 오른손 예측 결과 표시
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            this_action = action_right
            text = f'Right Hand: {this_action} ({confidence_right:.2f})'
            print(f'Right Hand: {this_action} ({confidence_right:.2f})')
            draw.text((10, 50), text, font=font, fill=(255, 0, 0, 0))
            img = np.array(img_pil)

        # 왼손 예측
        if len(seq_data_left) == seq_length:
            input_data_left = np.expand_dims(np.array(seq_data_left), axis=0)

            # feature_dim이 모델 기대 크기와 다르면 0으로 패딩
            current_feature_dim = input_data_left.shape[2]
            padding_dim = expected_feature_dim - current_feature_dim
            if padding_dim > 0:
                input_data_left = np.pad(input_data_left, ((0, 0), (0, 0), (0, padding_dim)), 'constant')

            # 예측 수행
            y_pred_left = best_left_model.predict(input_data_left).squeeze()

            # 예측 결과에 따라 동작 결정
            action_idx = np.argmax(y_pred_left)
            action_left = actions_left[action_idx]
            confidence_left = y_pred_left[action_idx]

            # 화면에 왼손 예측 결과 표시
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            this_action = action_left
            text = f'Left Hand: {this_action} ({confidence_left:.2f})'
            draw.text((10, 100), text, font=font, fill=(0, 255, 0, 0))
            print(f'Left Hand: {this_action} ({confidence_left:.2f})')
            img = np.array(img_pil)

    # 결과 영상 보여주기
    cv2.imshow('Hand Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()


