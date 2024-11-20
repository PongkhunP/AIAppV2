import cv2
import base64
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('process_image')
@socketio.on('process_image')
def handle_image(data):
    # Decode the base64 image
    image_data = data['image'].split(',')[1]  # Remove the 'data:image/jpeg;base64,' part
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Resize the image to reduce its size
    frame = cv2.resize(frame, (320, 240))  # Example size: 320x240 pixels

    # Process the frame (e.g., apply a model or draw shapes)
    processed_frame = cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)

    # Encode the processed frame to base64 for sending back to the frontend
    processed_frame = cv2.resize(processed_frame, (320, 240))
    _, buffer = cv2.imencode('.png', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    emit('processed_image', {'image': processed_frame_base64})


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
