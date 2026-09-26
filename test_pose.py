import cv2
import mediapipe as mp
import numpy as np

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)

def calculate_angle(a, b, c):
    """
    a, b, c 是三個點，每個點是 (x, y)
    計算 b 點的夾角，單位是度
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,     # 如果覺得卡頓，可以改成 0
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 開啟鏡頭
cap = cv2.VideoCapture(0)

# 改成可調整大小的窗口（解決最大化時灰色邊框的問題）
cv2.namedWindow("PhysioCare AI", cv2.WINDOW_NORMAL)

# 初始化全屏狀態
is_fullscreen = False

# 建議降低解析度以提升流暢度（可選）
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 鏡像翻轉，讓畫面像照鏡子一樣
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    h, w, c = frame.shape

    if results.pose_landmarks:
        # 1. 繪製 MediaPipe 預設的 3D 骨架連線
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # 2. 獲取所有關節點
        landmarks = results.pose_landmarks.landmark
        
        # 3. 提取左腳的關鍵點
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # 4. 將相對座標轉換為畫面的實際像素座標
        hip_x, hip_y = int(left_hip.x * w), int(left_hip.y * h)
        knee_x, knee_y = int(left_knee.x * w), int(left_knee.y * h)
        ankle_x, ankle_y = int(left_ankle.x * w), int(left_ankle.y * h)

        # 5. 畫出三個點 (方便視覺確認)
        cv2.circle(frame, (hip_x, hip_y), 10, (0, 0, 255), cv2.FILLED)    # 紅點：髖
        cv2.circle(frame, (knee_x, knee_y), 10, (0, 255, 0), cv2.FILLED)  # 綠點：膝
        cv2.circle(frame, (ankle_x, ankle_y), 10, (255, 0, 0), cv2.FILLED)# 藍點：踝

        # 6. 計算左膝角度 (髖-膝-踝)
        left_knee_angle = calculate_angle(
            (hip_x, hip_y), 
            (knee_x, knee_y), 
            (ankle_x, ankle_y)
        )

        # 7. 將角度顯示在畫面上（放大版，放在螢幕上方中央）
        text = f"Knee: {int(left_knee_angle)} deg"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0] 
        text_x = (w - text_size[0]) // 2
        text_y = 80
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5) # 黃色大字

        # 8. 判斷深蹲狀態
        if left_knee_angle > 150:
            status = "STANDING"
            color = (0, 255, 0) # 綠色
        elif 90 <= left_knee_angle <= 150:
            status = "SQUATTING"
            color = (0, 255, 255) # 黃色
        else:
            status = "DEEP SQUAT"
            color = (0, 0, 255) # 紅色

        # 9. 將狀態顯示在畫面下方
        cv2.putText(frame, f"Status: {status}", (50, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    cv2.imshow("PhysioCare AI", frame)
    
    # 按 'q' 或 'Esc' 退出
    # 按鍵處理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q 或 Esc 退出
        break
    elif key == ord('f'):             # f 鍵切換全屏
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty("PhysioCare AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("PhysioCare AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PhysioCare AI", 800, 600)

cap.release()
cv2.destroyAllWindows()