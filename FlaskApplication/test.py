import time
import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO("best.pt")

# Open the webcam (use `0` for default webcam)
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30  # Default to 30 FPS if unknown

timestamp = time.strftime("%Y%m%d-%H%M%S")
output_filename = f"output_{timestamp}.avi"

# Define the codec and create VideoWriter object
# out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Class Names (Ensure they match your model's training classes)
CLASS_RIDER = "rider"
CLASS_HELMET = "helmet"
CLASS_NO_HELMET = "no-helmet"

# Define colors for each class
COLOR_RIDER = (0, 255, 255)    # Yellow for Rider
COLOR_HELMET = (0, 255, 0)     # Green for Helmet
COLOR_NO_HELMET = (0, 0, 255)  # Red for No Helmet

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model(frame, imgsz=640, conf=0.8)

    # Extract detection information
    detections = results[0].boxes  # Get the bounding boxes and class info

    # Variables to track rider and helmet status
    is_rider_detected = False
    helmet_detected = False
    no_helmet_detected = False

    # Loop through detections and analyze conditions
    for detection in detections:
        class_id = int(detection.cls)  # Class index
        class_name = model.names[class_id]  # Get class name from the model
        bbox = detection.xyxy[0].tolist()  # Get bounding box coordinates (x1, y1, x2, y2)
        confidence = detection.conf.tolist()[0]  # Detection confidence score
        print(f"Detected {class_name} with confidence {confidence} at {bbox}")  # Debug information
        x1, y1, x2, y2 = map(int, bbox)

        # Check for rider
        if class_name == CLASS_RIDER:
            is_rider_detected = True
            label = "Rider"
            color = COLOR_RIDER
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check for helmet
        elif class_name == CLASS_HELMET:
            helmet_detected = True
            label = "Helmet"
            color = COLOR_HELMET
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check for no helmet
        elif class_name == CLASS_NO_HELMET:
            no_helmet_detected = True
            label = "No-Helmet"
            color = COLOR_NO_HELMET
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Check the conditions after processing all detections
    if is_rider_detected:
        if helmet_detected and not no_helmet_detected:
            cv2.putText(frame, "Riding with Helmet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_HELMET, 2)
        elif no_helmet_detected:
            cv2.putText(frame, "Riding without Helmet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NO_HELMET, 2)
        else:
            cv2.putText(frame, "Riding Status: Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No Rider Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the frame to the video file
    out.write(frame)

    # Display the frame with detections
    cv2.imshow("Real-Time Detection", frame)

    # Exit the video processing on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and VideoWriter objects, and close OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
