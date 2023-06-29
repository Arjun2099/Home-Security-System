from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle


faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings["arr_1"]

encoder = LabelEncoder()
encoder.fit(Y)
encoder.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(
    faces_embeddings["arr_0"], Y, shuffle=True, test_size=0.2, random_state=42
)

model = SVC(kernel="linear", probability=True)
# model = pickle.load(open("svm_model_160x160.pkl", "rb"))
model.fit(X_train, Y_train)

ypreds_train = model.predict(X_test)
ypreds_test = model.predict(X_test)


print(accuracy_score(Y_test, ypreds_test))
