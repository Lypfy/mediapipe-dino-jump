import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions, vision

pyautogui.PAUSE = 0.0

base_options = BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

# landmarks: x, y, z (0 -> 1)
def draw_hand(image, landmarks):
    h, w, _ = image.shape
    for p1, p2 in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
        x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

def get_index_finger_gesture(landmarks):
    """
    Nhận diện hành động dựa vào ngón trỏ
    """
    finger_base = landmarks[5]    # Khớp gốc ngón tay
    mid_joint_1 = landmarks[6]    # Khớp giữa dưới
    mid_joint_2 = landmarks[7]    # Khớp giữa trên
    finger_tip = landmarks[8]     # Đầu ngón trỏ

    # chọn chiều dài chuẩn: base -> joint 1
    base_segment_len = ((mid_joint_1.x - finger_base.x) ** 2 + (mid_joint_1.y - finger_base.y) ** 2) ** 0.5

    # tip -> base
    tip_to_base_dist = ((finger_tip.x - finger_base.x) ** 2 + (finger_tip.y - finger_base.y) ** 2) ** 0.5

    # ngón trỏ mở
    if finger_tip.y < mid_joint_1.y and finger_tip.y < mid_joint_2.y and tip_to_base_dist > 2 * base_segment_len:
        return "JUMP"

    # ngón trỏ đóng 
    if finger_tip.y > mid_joint_1.y or tip_to_base_dist < 0.1 * base_segment_len:
        return "DUCK"
    
    else:
         return "RUNNING"

def count_fingers(landmarks, handedness_label="Right"):
    fingers = []
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness_label == "Right":
        fingers.append(thumb_tip.x < thumb_ip.x)
    else:
        fingers.append(thumb_tip.x > thumb_ip.x)
    for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

    # [ngon cai, ngon tro, ngon giua, ngon ap ut, ngon ut]
    # true/false = mo/dong
    return fingers

cap = cv2.VideoCapture(0)
window_name = "Controller"

prev_state = "RUNNING"
STATE_COLORS = {
    "JUMP": (0, 255, 0),
    "DUCK": (0, 140, 255),
    "RUNNING": (255, 180, 0),
}

while True:
    returnval, frame = cap.read()
    if not returnval:
        print("Không kết nối được với webcam")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    current_state = "RUNNING"
    if result.hand_landmarks:
        hand_lms = result.hand_landmarks[0]
        draw_hand(frame, hand_lms)
        current_state = get_index_finger_gesture(hand_lms)

    if current_state != prev_state:
        if prev_state == "DUCK":
            pyautogui.keyUp('down')

        if current_state == "JUMP":
            pyautogui.press('space')
        elif current_state == "DUCK":
            pyautogui.keyDown('down')

        prev_state = current_state

    status_color = STATE_COLORS.get(current_state, (128, 128, 128))
    cv2.rectangle(frame, (15, 15), (200, 95), (30, 30, 30), -1)
    cv2.rectangle(frame, (15, 15), (200, 95), status_color, 2)
    cv2.putText(frame, f"STATUS: {current_state}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Action: {current_state}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

if prev_state == "DUCK":
    pyautogui.keyUp('down')

cap.release()
cv2.destroyAllWindows()
detector.close()
