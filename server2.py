import cv2
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import RPi.GPIO as GPIO
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Configure email settings
SMTP_SERVER = 'your_smtp_server_address'
SMTP_PORT = 587
EMAIL_USERNAME = 'your_email_username'
EMAIL_PASSWORD = 'your_email_password'
EMAIL_RECEIVER = 'receiver_email_address'

# IR sensor configuration
IR_SENSOR_PIN = 18

# Video streaming configuration
STREAMING_PORT = 8080

subjects = os.listdir('./faces/training-data')

def send_email_with_image(image):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USERNAME
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = 'Intruder Alert'

    text = MIMEText("An intruder has been detected. Please find the attached image.")
    msg.attach(text)

    image_data = cv2.imencode('.jpg', image)[1].tostring()
    image_attachment = MIMEImage(image_data)
    msg.attach(image_attachment)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USERNAME, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", str(e))

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y + w, x:x + h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        label = subjects.index(dir_name)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)

            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = detect_face(image)

            if face is not None:
                labels.append(label)
                faces.append(face)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("faces/training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img

def detect_intruder(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        label, confidence = face_recognizer.predict(face)

        if confidence < 50:
            send_email_with_image(frame)
            return True

    return False

class VideoFeedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            video_capture = cv2.VideoCapture(0)

            try:
                while True:
                    ret, frame = video_capture.read()

                    if detect_intruder(frame):
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', len(img_encoded))
                        self.end_headers()
                        self.wfile.write(img_encoded)
                        self.wfile.write(b'\r\n')

            except KeyboardInterrupt:
                pass

            video_capture.release()
            cv2.destroyAllWindows()

        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('', STREAMING_PORT)
    httpd = HTTPServer(server_address, VideoFeedHandler)
    print("Server started")
    httpd.serve_forever()

# Set up GPIO for IR sensor
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_SENSOR_PIN, GPIO.IN)

# Start the video streaming server when an intruder is detected
if GPIO.input(IR_SENSOR_PIN):
    streaming_thread = threading.Thread(target=run_server)
    streaming_thread.daemon = True
    streaming_thread.start()

try:
    while True:
        pass
except KeyboardInterrupt:
    pass

# Clean up GPIO
GPIO.cleanup()
