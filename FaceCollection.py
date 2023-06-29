import cv2 as cv
import os
import numpy as np
import keyboard


class FaceCollection:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.harr_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.cap = cv.VideoCapture(1)
        self.faces = []
        self.names = []
        self.count = 0

    def extract_face(self, filename):
        img = cv.imread(filename)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.harr_cascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            face = img[y : y + h, x : x + w]
            face = cv.resize(face, self.target_size)
            return face

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                face = self.extract_face(os.path.join(dir, im_name))
                FACES.append(face)
            except Exception as e:
                print(e)

        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]

            self.faces.extend(FACES)
            self.names.extend(labels)

        return np.asarray(self.faces), np.asarray(self.names)

    def save_faces(self, faces, names):
        np.savez_compressed(
            "faces_embeddings_done_4classes.npz", faces=faces, names=names
        )

    def collect_faces(self):
        face_name = input("Enter name: ")
        os.mkdir("./faces/train_data/" + face_name)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # gray_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            try:
                if self.count >= 1000:
                    break
                # faces = self.harr_cascade.detectMultiScale(gray_img, 1.3, 5)

                # for x, y, w, h in faces:
                #     img = frame[y : y + h, x : x + w]
                #     img = cv.resize(img, self.target_size)
                #     # img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

                #     file_name = (
                #         "faces/train_data/" + face_name + "/" + str(self.count) + ".jpg"
                #     )
                #     cv.imwrite(file_name, img)
                frame = cv.resize(frame, (640, 480))
                file_name = (
                    "faces/train_data/" + face_name + "/" + str(self.count) + ".jpg"
                )
                cv.imwrite(file_name, frame)
                self.count += 1

                cv.imshow("frame", frame)
                cv.waitKey(1)
            except Exception as e:
                print(e)

            if keyboard.is_pressed("q"):
                break

        self.cap.release()
        cv.destroyAllWindows()


obj = FaceCollection("faces/train_data")
obj.collect_faces()
