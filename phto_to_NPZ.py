from keras_facenet import FaceNet
import cv2 as cv
import os
import numpy as np
import keyboard


class Photo_to_npx:
    def __init__(self, train_dir) -> None:
        self.target_size = (160, 160)
        self.harr_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.embedder = FaceNet()
        self.faces = []
        self.names = []
        self.train_dir = train_dir

    def extract_face(self, filename):
        img = cv.imread(filename)
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        faces = self.harr_cascade.detectMultiScale(gray_img, 1.3, 5)
        if len(faces) > 0:
            for x, y, w, h in faces:
                face = img[y : y + h, x : x + w]
                face = cv.resize(face, self.target_size)
                return face
        else:
            raise Exception("No face found in image")

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            # print(f"Loading {im_name} from {dir}")
            try:
                face = self.extract_face(os.path.join(dir, im_name))
                FACES.append(face)
            except Exception as e:
                print(e)

        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.train_dir):
            path = os.path.join(self.train_dir, sub_dir)
            FACES = self.load_faces(path)
            # print(f"Loaded {len(FACES)} faces from {sub_dir}")
            labels = [sub_dir for _ in range(len(FACES))]
            # print(f"Loaded {labels} labels from {sub_dir}")
            self.faces.extend(FACES)
            self.names.extend(labels)

        return np.asarray(self.faces), np.asarray(self.names)

    def get_embeddings(self, faces):
        faces_img = faces.astype(np.float32)
        faces_img = (faces_img - faces_img.mean()) / faces_img.std()
        faces_img = np.expand_dims(faces_img, axis=0)
        yhat = self.embedder.embeddings(faces_img)
        return yhat[0]

    def save_faces(self):
        embadded_x = []
        for img in self.faces:
            embadded_x.append(self.get_embeddings(img))
        embadded_x = np.asarray(embadded_x)

        np.savez_compressed(
            "faces_embeddings_done_4classes.npz", embadded_x, self.names
        )


train_dir = r"./faces/train_data/"
photo_to_npx = Photo_to_npx(train_dir)
faces, names = photo_to_npx.load_classes()
photo_to_npx.save_faces()
