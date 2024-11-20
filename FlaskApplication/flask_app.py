import base64
from email import utils
import io
import threading
from tkinter import Image
from flask import Flask, redirect, render_template, Response,jsonify,request, send_from_directory,session, url_for
from flask_socketio import SocketIO, emit
#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model
from dotenv import load_dotenv
from flask_wtf import FlaskForm
from flask_cors import CORS

import mariadb
import numpy as np
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os

from yoloVid import video_detection_frame
from storage import get_db_connection
# Required to run the YOLOv8 model
import cv2
# YOLO_Video is the python file which contains the code for our object detection model
#Video Detection is the Function which performs Object Detection on Input Video

load_dotenv()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://127.0.0.1:5000")

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
# print(os.getenv('MARIA_DB_PATH'))

with app.app_context():
    from yoloVid import video_detection
# app.config['UPLOAD_FOLDER'] = 'static/files'
is_running = False

#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()]) 
    submit = SubmitField("Run")


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x,True)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

def generate_frames_web(path_x, is_run,confidence_lv):
    yolo_output = video_detection(path_x,is_run,confidenceLv=confidence_lv)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        
# @socketio.on('start_video_stream')
# def start_video_stream(path_x, is_run,confidence_lv):
#     # Extract parameters from the client request
#     video_path = path_x
#     confidence_level = confidence_lv
#     is_running = is_run

#     # Start video detection
#     video_detection(is_running)

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('home.html')

@app.route('/gallery')
def gallery():
    session.clear()
    return render_template('gallery.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    if 'is_running' not in session:
        session['is_running'] = False  # default value
    return render_template('index.html', is_running=session['is_running'])

@socketio.on('frame')
def handle_frame(data):
    # Decode the incoming image
    encoded_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame using YOLO
    yolo_output = video_detection_frame(frame)

    # Encode the processed frame to JPEG and send it back
    _, buffer = cv2.imencode('.jpg', yolo_output)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    emit('processed_frame', {'image': frame_base64})

@app.route('/set_selected_webcam', methods=['POST'])
def set_selected_webcam():
    data = request.get_json()
    webcam_id = data.get("webcam_id")

    if webcam_id is not None:
        session['selected_webcam'] = int(webcam_id)
        return jsonify({"message": "Webcam updated successfully"}), 200
    else:
        return jsonify({"error": "Invalid webcam ID"}), 400

@app.route('/set_conf_lv', methods=['POST'])
def set_conf_level():
    data = request.get_json()
    conf_lv = data.get("conf_lv")

    if conf_lv is not None:
        try:
            session['selected_conf_lv'] = float(conf_lv)
            return jsonify({"message": "Confidence level updated successfully"}), 200
        except ValueError:
            return jsonify({"error": "Invalid confidence level format"}), 400
    else:
        return jsonify({"error": "Confidence level not provided"}), 400


@app.route('/get_all_conf_lv_available', methods=['GET'])
def get_all_conf_lv_available():
    conf_lv = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    return jsonify({"conf_lv": conf_lv}), 200


 #//
@app.route('/get_available_webcams', methods=['GET'])
def get_available_webcams():
    webcams = []
    for i in range(5):  # Check the first 10 indices for connected webcams
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            webcams.append({"id": i, "name": f"Webcam {i}"})
            cap.release()
    return jsonify(webcams), 200



@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    # Toggle the 'is_running' value in the session
    session['is_running'] = not session.get('is_running', False)
    return redirect(url_for('webcam'))

@app.route('/get_capture_image', methods=['GET'])
def get_image():
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            # Execute a query to get all image data from the `footage` table
            cur.execute("SELECT image_url, date FROM footage")
            rows = cur.fetchall()  # Fetch all rows from the result

            # Convert the result into a list of dictionaries
            images = [{"image_url": row[0], "date": row[1]} for row in rows]

            return jsonify(images), 200
        except mariadb.Error as e:
            return jsonify({"error": f"Failed to retrieve images: {e}"}), 500
        finally:
            conn.close()
    else:
        return jsonify({"error": "Failed to connect to the database."}), 500

@app.route('/webapp')
def webapp():
    is_running = session.get('is_running', False)
    selected_webcam = session.get('selected_webcam', 0)  # Default to webcam 0
    confidence_level = session.get('selected_conf_lv', 0.7)
    if is_running:
        # Use the selected webcam index
        return Response(start_video_stream(0,False,0.8), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return send_from_directory('static/images', 'camera-off.svg')

@app.route('/video_feed')
def video_feed():
    is_running = session.get('is_running', False)
    if is_running:
        return Response(video_detection(is_running=is_running), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return send_from_directory('static/images', 'camera-off.svg')

@socketio.on('start_video_stream')
def start_video_stream(data):
    path_x = data.get('path_x', 0)  # Default webcam index
    is_running = data.get('is_running', True)
    confidence_lv = data.get('confidence_lv', 0.7)

    def video_thread():
        video_feed()

    thread = threading.Thread(target=video_thread)
    thread.daemon = True
    thread.start()


@socketio.on('stop_video_stream')
def stop_video_stream():
    global is_running
    is_running = False

@socketio.on('connect')
def connect():
    emit('my response', {'data': 'Connected'})


if __name__ == "__main__":
    socketio.run(app, debug=True)