from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)
harr_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", "rb"))

cap = cv.VideoCapture(1)

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = harr_cascade.detectMultiScale(gray_img, 1.3, 5)

    try:
        for x, y, w, h in faces:
            img = rgb_img[y : y + h, x : x + w]
            img = cv.resize(img, (160, 160))
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            print(face_name)
            print(encoder.inverse_transform(face_name))
            final_name = encoder.inverse_transform(face_name)[0]
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(
                frame,
                str(final_name),
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3,
                cv.LINE_AA,
            )

    except Exception as e:
        print(e)

    cv.imshow("frame", frame)
    if cv.waitKey(1) & ord("q") == 27:
        break

cap.release()
cv.destroyAllWindows()
