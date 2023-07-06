import os
import time
import smtplib
import datetime
import cv2 as cv
import subprocess
import http.server
import numpy as np
import threading
import time
import socketserver
import multiprocessing
from email import encoders
from collections import deque
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client 
from dotenv import load_dotenv



# Webcam Video Stream
class VideoStream:
    def __init__(self):
        self.video_capture = cv.VideoCapture(0)

    def get_frame(self):
        ret, frame = self.video_capture.read()
        ret, jpeg = cv.imencode('.jpg', frame)
        return jpeg.tobytes()


# HTTP Request Handler
class VideoStreamHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            video_stream = VideoStream()

            while True:
                frame = video_stream.get_frame()

                self.send_frame(frame)

    def send_frame(self, frame):
        self.wfile.write(b'--frame\r\n')
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Content-length', len(frame))
        self.end_headers()
        self.wfile.write(frame)
        self.wfile.write(b'\r\n')


# Run HTTP Server
def run_server():
    address = ('', 8000)
    http_server = socketserver.TCPServer(address, VideoStreamHandler)
    print('Server running on http://192.168.37.89:8000/video_feed')

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("keyboard interrupt occured")

    http_server.server_close()
    print('Server stopped')


class SmartDetection:
    def __init__(self):
        load_dotenv() 
        self.cap = cv.VideoCapture(0)
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))
        self.foog = cv.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=100, history=2000)
        self.status = False
        self.patience = 7
        self.detection_thresh = 15
        self.initial_time = None
        self.de = deque([False] * self.detection_thresh, maxlen=self.detection_thresh)
        self.fps = 0
        self.frame_counter = 0
        self.start_time = time.time()
        self.SMTP_SERVER = os.getenv('SMTP_SERVER')
        self.SMTP_PORT = int(os.getenv('SMTP_PORT'))
        self.EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
        self.EMAIL_RECEIVER = os.getenv('EMAIL_RECEIVER')
        self.account_sid = os.getenv('account_sid')
        self.auth_token = os.getenv('auth_token')
        self.trial_Phone_num = os.getenv('trial_num')
        self.client_Phone_num = os.getenv('client_num')
        self.client = Client(self.account_sid, self.auth_token)
        self.message = None
        self.msg = MIMEMultipart()
        self.msg['From'] = self.EMAIL_USERNAME
        self.msg['To'] = self.EMAIL_RECEIVER
        self.msg['Subject'] = 'Intruder Alert'
        self.confident = 0.7
        self.vid_save_dir = r"output"
        self.video_flag = True
        self.val_flag = True
        self.video_link = r"http://192.168.37.89:8000/video_feed"

    def is_person_present(self, frame, thresh=1100):
        fgmask = self.foog.apply(frame)
        ret, fgmask = cv.threshold(fgmask, 250, 255, cv.THRESH_BINARY)
        fgmask = cv.dilate(fgmask, None, iterations=4)
        contours, hierarchy = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours and cv.contourArea(max(contours, key=cv.contourArea)) > thresh:
            cnt = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(cnt)
            return True, frame
        else:
            return False, frame

    def send_message(self, body,flag,attachment):
        client = Client(self.account_sid, self.auth_token)
        sms_message = client.messages.create(
            from_ = self.trial_Phone_num,
            body = body,
            to = self.client_Phone_num,
        )
        msg = MIMEMultipart()
        msg['From'] = self.EMAIL_USERNAME
        msg['To'] = self.EMAIL_RECEIVER
        msg['Subject'] = "Intruder Alert"
        msg.attach(MIMEText(body, 'plain'))
        print("SMS sent successfully!")
        print()
        print("Sending email...")

        if flag:
            with open(attachment, 'rb') as file:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(file.read())
                encoders.encode_base64(attachment)
                attachment.add_header('Content-Disposition', 'attachment; filename={attachment}')
                msg.attach(attachment)

                try:
                    server = smtplib.SMTP(self.SMTP_SERVER, self.SMTP_PORT)
                    server.starttls()
                    server.login(self.EMAIL_USERNAME, self.EMAIL_PASSWORD)

                    server.sendmail(self.EMAIL_USERNAME, self.EMAIL_RECEIVER, msg.as_string())
                    server.quit()
                    print("Email sent successfully!")
                    self.video_flag = False
                except Exception as e:
                    print("Failed to send email:", str(e))

    def main(self):
        if self.confident >= 0.7:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                detected, annotated_image = self.is_person_present(frame)

                self.de.appendleft(detected)


                if sum(self.de) == self.detection_thresh and not self.status:
                    print("Movement detected !!!")
                    print("Starting video Capture...")

                    self.status = True
                    entry_time =  datetime.datetime.now().strftime("%A, %I:%M:%S %p %d %B %Y")
                    video_id = datetime.datetime.now().strftime("%A,%I-%M-%S%p%d%B%Y")
                    vid_name = os.path.join(self.vid_save_dir, f"{video_id}.mp4")
                    print(f"video name: {vid_name}")
                    out = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'mp4v'), 15.0, (self.width, self.height))
                    print()
                    print("Video saved")
                    print()
                    print("***********************************************************")
                    print()
                    print("Sending SMS & email alert...")
                    attachment = vid_name

                if self.status and not detected:
                    if sum(self.de) > (self.detection_thresh / 2):
                        if self.initial_time is None:
                            self.initial_time = time.time()
                    elif self.initial_time is not None:
                        if time.time() - self.initial_time >= self.patience:
                            self.status = False
                            exit_time = datetime.datetime.now().strftime("%A, %I:%M:%S %p %d %B %Y")
                            out.release()
                            self.initial_time = None
                            body = f"Alert: A person entered the room at {entry_time} and left the room at {exit_time},click here for live video {self.video_link}"
                            self.cap.release()
                            cv.destroyAllWindows()
                            self.send_message(body,self.video_flag,attachment)
                            print()
                            print("***********************************************************")
                            print()
                            print("Starting the live server...")
                            if os.path.exists(vid_name):
                                # Delete the file
                                os.remove(vid_name)
                            else:
                                print("Video not saved")
                            run_server()

                if self.status:
                    out.write(annotated_image)

                self.frame_counter += 1
                self.fps = (self.frame_counter / (time.time() - self.start_time))

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            print("No intruders")

if __name__ == "__main__":
    print("***********************************************************")
    print()
    print("Starting movement detection...")
    obj = SmartDetection()
    obj.main()
