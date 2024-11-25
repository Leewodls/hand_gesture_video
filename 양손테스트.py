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

# 학습된 모델 로드
model = tf.keras.models.load_model('best_both_model_video_c.keras')

# 동작 리스트 정의 (모델 학습 시 사용한 클래스와 동일하게)
actions = ['가렵다', '가스', '가슴', '각목', '갇히다', '감금', '감전', '강남구', '강동구', '강북구', '강서구', 
           '강풍', '개', '거실', '결박', '경운기', '경찰차', '계곡', '계단', '고속도로', '고압전선', '고열', '고장', 
           '부러지다', '골절', '탈골', '공사장', '공원', '놀이터', '공장', '관악구', '광진구', '교통사고', '구급대', 
           '구급대원', '구급차', '구로구', '구해주세요', '금요일', '금천구', '급류', '기절', '기절하다', '깔리다', 
           '끓는물', '남자친구', '남편', '남학생', '납치', '낯선남자', '낯선사람', '내년', '노원구', '논', '농약', 
           '누전', '누출', '눈', '다리', '다음', '대문앞', '도둑', '절도', '도로', '도봉구', '독버섯', '독사', 
           '동대문구', '동작구', '두드러기생기다', '딸', '떨어지다', '쓰러지다', '뜨거운물', '마당', '마포구', 
           '말려주세요', '말벌', '맹견', '멧돼지', '목요일', '무너지다', '붕괴', '문틈', '밑에', '아래', '바다', 
           '반점생기다', '발', '발가락', '발목', '발작', '방망이', '밭', '배고프다', '뱀', '범람', '벼락', '병원', 
           '보건소', '보내주세요(경찰)', '보내주세요(구급차)', '복통', '부엌', '불', '불나다', '붕대', '비닐하우스', 
           '비상약', '빌라', '뼈', '사이', '산', '살충제', '살해', '서대문구', '서랍', '서울시', '서초구', '선반', 
           '선생님', '성동구', '성북구', '성폭행', '소방관', '소방차', '소화기', '소화전', '손', '손가락', '손목', 
           '송파구', '수영장', '술취한 사람', '시청', '심장마비', '아기', '아이들', '어린이', '아내', '아저씨', 
           '아줌마', '아파트', '인대', '안방', '알려주세요', '앞집', '약국', '약사', '양천구', '엘리베이터', '여자친구', 
           '여학생', '연락해주세요', '열', '열나다', '열어주세요', '엽총', '영등포구', '옆집', '옆집 아저씨', '옆집 할아버지', 
           '옆집사람', '오늘', '오른쪽', '옥상', '올해', '왼쪽', '욕실', '용산구', '우리집', '운동장', '위에', '위협', 
           '윗집', '윗집사람', '유리', '유치원', '유치원 버스', '은평구', '음식물', '응급대원', '응급처리', '의사', 
           '이물질', '이번', '이상한사람', '이웃집', '일요일', '임산부', '임신한아내', '자동차', '자살', '자상', 
           '작년', '작은방', '장난감', '장단지', '절단', '제초제', '조난', '종로구', '중구', '중랑구', '지혈대', 
           '진통제', '질식', '집', '집단폭행', '차밖', '차안', '창문', '창백하다', '체온계', '총', '추락', '축사', 
           '출산', '출혈', '피나다', '친구', '침수', '칼', '코', '택시', '토요일', '토하다', '통학버스', '트랙터', 
           '트럭', '파도', '파편', '팔', '팔꿈치', '폭발', '폭탄', '폭우', '폭행', '학교', '학생', '함몰되다', 
           '해(연)', '해독제', '해열제', '현관', '현관앞', '협박', '호흡곤란', '홍수', '화상', '화약', '화재']

# 시퀀스 길이 고정 (모델 학습 시 사용한 값과 동일하게)
seq_length = 60

# 한글 폰트 경로 설정
font_path = "/Users/ijaein/Desktop/졸업프로젝트/ex_meme/Noto_Sans_KR/NotoSansKR-VariableFont_wght.ttf"
font = ImageFont.truetype(font_path, 30)

# 웹캠 열기
cap = cv2.VideoCapture(1)

seq_data = []  # 실시간 데이터를 저장할 리스트

# 이전 예측 동작 저장 변수 및 타이머
previous_action = None
start_time = 0
current_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks is not None:
        joint = np.zeros((126 * 2,))  # 양손의 랜드마크 데이터 (21*3*2)
        hand_count = 0  # 손의 개수를 추적

        for hand_landmarks in result.multi_hand_landmarks:
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[hand_count * 21 * 3 + j * 3] = lm.x
                joint[hand_count * 21 * 3 + j * 3 + 1] = lm.y
                joint[hand_count * 21 * 3 + j * 3 + 2] = lm.z
            hand_count += 1

            # 양손의 랜드마크 그리기
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 최대 두 손만 처리하도록 설정
            if hand_count == 2:
                break

        # 실시간 데이터 추가
        seq_data.append(joint)

        # 시퀀스 길이 유지 (최대 seq_length 프레임 저장)
        if len(seq_data) > seq_length:
            seq_data.pop(0)

        # 시퀀스가 충분히 쌓였을 때 예측
        if len(seq_data) == seq_length:
            input_data = np.expand_dims(np.array(seq_data), axis=0)  # 모델 입력 형식에 맞게 차원 추가
            y_pred = model.predict(input_data).squeeze()

            # 예측 결과에 따라 동작 결정
            action_idx = np.argmax(y_pred)

            # 예측 인덱스가 유효한지 확인
            if action_idx < len(actions):
                action = actions[action_idx]
                confidence = y_pred[action_idx]

                # 예측 결과 출력
                print(f'예측된 동작: {action} ({confidence:.2f})')

                # 같은 동작이 1초 동안 유지되었을 때만 자막 업데이트
                if action == previous_action:
                    if time.time() - start_time >= 1:
                        current_action = action
                else:
                    previous_action = action
                    start_time = time.time()
                    current_action = None
            else:
                current_action = None

            # 화면에 예측 결과 표시
            if current_action:
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                text = f'{current_action} ({confidence:.2f})'
                draw.text((10, 150), text, font=font, fill=(255, 0, 0, 0))
                img = np.array(img_pil)

    # 결과 영상 보여주기
    cv2.imshow('Hand Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()