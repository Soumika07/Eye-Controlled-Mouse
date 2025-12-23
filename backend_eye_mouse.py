import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

pyautogui.FAILSAFE = False

# -------------------------------
# CONFIG
# -------------------------------
EAR_THRESHOLD = 0.25
CLICK_COOLDOWN = 10
alpha = 0.12

# -------------------------------
# UTILITIES
# -------------------------------
def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def get_EAR(lm):
    p1, p4 = lm[33], lm[133]
    p2, p6 = lm[159], lm[145]
    p3, p5 = lm[158], lm[153]
    return (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

# -------------------------------
# INIT
# -------------------------------
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_w, screen_h = pyautogui.size()
smooth_x, smooth_y = screen_w / 2, screen_h / 2

user_locked = False
lock_time = None
cooldown = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        if not user_locked:
            user_locked = True
            lock_time = time.time()

        lm = results.multi_face_landmarks[0].landmark

        # LOCK MESSAGE
        if time.time() - lock_time < 2:
            cv2.rectangle(frame, (50, 100), (w - 50, 220), (0, 0, 0), -1)
            cv2.putText(frame, "EYES DETECTED & LOCKED",
                        (60, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        # IRIS CONTROL
        iris = lm[476]
        ix, iy = int(iris.x * w), int(iris.y * h)
        cv2.circle(frame, (ix, iy), 4, (0, 255, 0), -1)

        target_x = np.interp(ix, [w * 0.25, w * 0.75], [0, screen_w])
        target_y = np.interp(iy, [h * 0.35, h * 0.75], [0, screen_h])

        smooth_x += (target_x - smooth_x) * alpha
        smooth_y += (target_y - smooth_y) * alpha
        pyautogui.moveTo(smooth_x, smooth_y)

        # BLINK CLICK
        ear = get_EAR(lm)
        cv2.putText(frame, f'EAR: {ear:.3f}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        if ear < EAR_THRESHOLD and cooldown == 0:
            pyautogui.click()
            cooldown = CLICK_COOLDOWN

    else:
        # Face lost → unlock
        user_locked = False

    if cooldown > 0:
        cooldown -= 1

    cv2.imshow("Eye Controlled Mouse", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
