#!/usr/bin/env python
# coding: utf-8

# pre requistie libraries
# ('pip install ultralytics')
# ('pip install qrcode')
# ('pip install opencv-python-headless')
# ('pip install SQLAlchemy')
# ('pip install Pillow')

#database module
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    quantity = Column(Integer)
    status = Column(String)
    barcode = Column(String)

engine = create_engine('sqlite:///products.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)


#live feeding module

# import cv2
# import requests
# from ultralytics import YOLO
# import time

# # ESP32-CAM stream URL (update with your IP address)
# stream_url = 'http://192.168.1.104:81/stream'

# # Load YOLO model
# model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' or your custom model

# # Open the video stream
# cap = cv2.VideoCapture(stream_url)

# if not cap.isOpened():
#     print("❌ Error: Could not open ESP32-CAM stream.")
#     exit()

# # Wait for the camera to stabilize (optional, avoids blurry frames)
# time.sleep(2)

# # Capture one frame
# ret, frame = cap.read()

# if ret:
#     # Save frame as image
#     image_path = "frame.jpg"
#     cv2.imwrite(image_path, frame)
#     print("✅ Frame captured and saved.")

#     # Run YOLOv8 object detection
#     results = model.predict(source=image_path, show=True, save=True)
#     print("✅ Detection complete.")

#     # Send image to Flask server (update IP accordingly)
#     try:
#         with open(image_path, "rb") as img_file:
#             response = requests.post(
#                 "http://192.168.0.132:5000/upload",
#                 files={"image": img_file}
#             )
#         if response.status_code == 200:
#             print("✅ Image successfully uploaded to Flask server.")
#         else:
#             print(f"❌ Upload failed: {response.status_code} - {response.text}")
#     except Exception as e:
#         print("❌ Error during image upload:", e)

# else:
#     print("❌ Failed to capture frame from stream.")

# # Release the camera stream
# cap.release()


#static taking
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from collections import Counter

# Detection and Database update
image_path = "bottle.jpeg"  # Update with your image path
model = YOLO('yolov8m.pt')  # Use your model path

# Read image
img = cv2.imread(image_path)

if img is None:
    print("❌ Error loading image.")
else:
    # Run detection
    results = model(img)

    # Get class labels and box coordinates
    detected_classes = results[0].boxes.cls.tolist()
    detected_items = [model.names[int(cls)] for cls in detected_classes]

    if detected_items:
        detected_counts = dict(Counter(detected_items))
        print("🧾 Detected items & counts:", detected_counts)

        # Draw bounding boxes and labels
        for box, cls_idx in zip(results[0].boxes.xyxy.tolist(), detected_classes):
            label = model.names[int(cls_idx)]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

        # Save or show image
        cv2.imwrite("output.jpg", img)

        # Update the database if there are detected items
        session = Session()
        for name, count in detected_counts.items():
            product = session.query(Product).filter_by(name=name).first()
            if product:
                product.quantity += count
            else:
                product = Product(name=name, quantity=count, status="Normal", barcode=f"{name}_123")
                session.add(product)
        session.commit()
        session.close()

    else:
        print("❌ No objects detected in the image.")



def search_product_by_code(code):
    session = Session()
    product = session.query(Product).filter_by(barcode=code).first()
    session.close()
    if product:
        return {
            'name': product.name,
            'quantity': product.quantity,
            'status': product.status
        }
    else:
        return "Product not found."

# 🔍 Example search
search_product_by_code("bottle_123")


# In[17]:


import qrcode
from PIL import Image

def generate_qr_for_product(name):
    code = f"{name}_123"
    img = qrcode.make(code)
    img.save(f"{name}_qr.png")
    print(f"QR for {name} saved as {name}_qr.png")

# Example
generate_qr_for_product("bottle")


# In[18]:


import qrcode
from PIL import Image

def generate_qr_for_product(name):
    # Concatenate the product name with a unique identifier
    code = f"{name}_123"

    # Create a QRCode object with enhanced options for better readability
    qr = qrcode.QRCode(
        version=2,  # Higher version for a larger QR code
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # Higher error correction level
        box_size=10,  # Box size for each square in the QR code
        border=4,  # Border size
    )

    qr.add_data(code)
    qr.make(fit=True)

    # Generate the image for the QR code
    img = qr.make_image(fill='black', back_color='white')

    # Save the generated QR code as an image file
    img.save(f"{name}_qr.png")
    print(f"QR for {name} saved as {name}_qr.png")

    # Optionally display the QR code image
    img.show()

# Example: Generate QR for a product
generate_qr_for_product("bottle")

# Create an HTML file for the product page
product_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Page - Bottle</title>
</head>
<body>
    <header>
        <h1>Product: Bottle</h1>
    </header>
    <section>
        <img src="https://www.example.com/bottle_image.jpg" alt="Bottle Image" width="200"> <!-- Add your image URL here -->
        <p><strong>Price:</strong> $20.00</p>
        <p><strong>Description:</strong> A durable and reusable bottle for everyday use.</p>
        <p><strong>Product Code:</strong> bottle_123</p>
    </section>
    <footer>
        <p>&copy; 2025 YourStore.com</p>
    </footer>
</body>
</html>
"""

# Save the HTML to a file
with open("product_page.html", "w") as file:
    file.write(product_html)

print("Product page HTML file created!")
# 'pip install qrcode[pil]'
import qrcode

def generate_qr_for_product_page(product_url):
    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR code
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each box
        border=4,  # Border size
    )

    qr.add_data(product_url)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white')

    # Save the image
    img.save("/content/product_qr.png")
    print("QR Code for product page generated and saved as 'product_qr.png'")

# Example usage
product_url = "https://illustrious-medovik-dc72d2.netlify.app/"  # Replace with your URL
generate_qr_for_product_page(product_url)

from PIL import Image

# Open and display the QR code
img = Image.open("product_qr.png")
img.show()

#ml features-->
import pandas as pd
from ml_features import MLForecasting
sales_data = {
    'ds': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'y': [20, 23, 19, 25, 22, 26, 30, 28, 35, 40]  # Sample bottle sales
}
df = pd.DataFrame(sales_data)
ml_forecaster = MLForecasting()
forecast = ml_forecaster.demand_forecasting("Bottle", df)
print("📈 Forecasted Demand:")
print(forecast)

