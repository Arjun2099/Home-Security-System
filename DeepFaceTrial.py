import threading
import socket
import cv2
from deepface import DeepFace
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
imgCounter = 10
face_match = False
harr_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

reference_imgs = [cv2.imread(f'./training data/{image}') for image in os.listdir('./training data')]

def check_face(frame):
    global face_match
    global reference_img
    global imgCounter

    try:
        faces = harr_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (224, 224))

            for image in reference_imgs:
                result = DeepFace.verify(image.copy(), face)['verified']
                if result:
                    face_match = True
                    imgCounter += 1
                else:
                    face_match = False


    except ValueError:
        face_match = False


while True:

    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        if (counter/30) % 10 == 0:
            print(f"images counter val : {imgCounter}")
            if imgCounter <= 7:
                host = '192.168.138.89'
                port = 12345
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((host, port))
                        message = "Intruder Alert"
                        s.sendall(message.encode())
                except ConnectionRefusedError:
                    pass
            imgCounter = 0
        counter += 1

        if face_match:
            cv2.putText(frame, "Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Face Not Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
