import cv2
from ultralytics import YOLO

# ESP32-CAM stream URL
stream_url = 'http://192.168.172.131/1024x768.mjpeg'  # Replace with your ESP32-CAM IP

# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open stream.")
    exit()

# Read one frame from the stream
ret, frame = cap.read()

if ret:
    # Save frame as an image
    cv2.imwrite("frame.jpg", frame)
    print("Captured frame from stream.")

    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Or use your custom model

    # Run detection
    results = model.predict(source="frame.jpg", show=True, save=True)
    print("Detection complete.")
else:
    print("Failed to capture frame.")

# Release the stream
cap.release() 