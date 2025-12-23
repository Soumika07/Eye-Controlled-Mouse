import cv2
import mediapipe as mp
import time

def run_frontend():
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cam = cv2.VideoCapture(0)
    locked = False
    lock_time = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        h, w, _ = frame.shape

        if res.multi_face_landmarks and not locked:
            locked = True
            lock_time = time.time()

        if locked:
            cv2.rectangle(frame, (40, 120), (w - 40, 260), (0, 0, 0), -1)
            cv2.putText(frame, "EYES DETECTED & LOCKED",
                        (60, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 4)

            if time.time() - lock_time > 1.5:
                break

        cv2.imshow("Eye Lock Screen", frame)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_frontend()
