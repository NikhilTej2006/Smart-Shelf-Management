# Flask server (running on PC)
from flask import Flask, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Save or process image
    cv2.imwrite('latest.jpg', img)
    return 'Image received', 200

app.run(host='0.0.0.0', port=5000)


# app.py
# from flask import Flask, request
# import cv2
# import numpy as np

# app = Flask(__name__)

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     file = request.files['image']
#     img_np = np.frombuffer(file.read(), np.uint8)
#     img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

#     # Save or process image
#     cv2.imwrite('latest.jpg', img)
#     return 'Image received and saved.', 200

# app.run(host='0.0.0.0', port=5000)
