import cv2
import requests
import webbrowser

# URL for the video stream
VIDEO_STREAM_URL = 'http://<SERVER_IP_ADDRESS>:5000/video_feed'

# Email notification settings
EMAIL_RECEIVER = 'your_email_address'

def send_email_alert():
    # Implement your code to send an email alert using your preferred email library or service

def process_frame(frame):
    # Implement your code to process the frame (e.g., face detection)

    # If an intruder is detected, send an email alert
    send_email_alert()

def view_video_stream():
    # Open the browser to view the video stream
    webbrowser.open(VIDEO_STREAM_URL)

def main():
    # Start the video capture
    cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index if multiple cameras are connected

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        process_frame(frame)

        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    view_video_stream()
    main()
