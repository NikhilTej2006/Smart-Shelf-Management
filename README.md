Smart Shelf Management System
This project aims to automate shelf management using computer vision and IoT technologies. It tracks product quantities, detects objects using machine learning models (YOLOv5), and manages stock levels through an inventory system.

Features
Product recognition using YOLOv5 and ESP32-CAM

Real-time inventory tracking

Stock level prediction using time-series models (LSTM/Prophet)

Demand forecasting using machine learning algorithms (Random Forest, XGBoost)

Flask-based backend for data processing and API

Tech Stack
Computer Vision: YOLOv5, OpenCV

Machine Learning: TensorFlow, Keras, scikit-learn

Backend: Flask

Database: SQL-based (SQLite or MySQL)

IoT: ESP32-CAM for image capture

Data Prediction: LSTM, Prophet, Random Forest, XGBoost

Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/smart-shelf-management.git
cd smart-shelf-management
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Start the Flask Server
bash
Copy
Edit
cd server
python app.py
4. Set Up ESP32-CAM
Flash the ESP32-CAM with the appropriate firmware to capture images and send them to the Flask server.

5. Run Object Detection
Upload product images or stream live video from ESP32-CAM to detect and update inventory.

Project Structure
bash
Copy
Edit
smart-shelf-management/
│
├── model/                 # YOLOv5 and object detection code
├── server/                # Flask server and database integration
├── esp32/                 # ESP32-CAM firmware code
├── prediction/            # LSTM, Prophet, Random Forest models for prediction
└── README.md              # Project documentation
Future Scope
Integrate barcode scanning for faster product detection

Implement real-time notifications for low stock levels

Add predictive analytics for inventory management

License
This project is licensed under the MIT License.
