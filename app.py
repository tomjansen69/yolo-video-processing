import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2

# Load the YOLO model and class names
model = YOLO('models/medium.pt')
class_names = model.names

def process_video(input_path, output_path, progress_callback):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = class_names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        progress_callback(int(frame_count / total_frames * 100))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

st.title("YOLO Video Processing App")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the uploaded file to the temporary directory with the correct extension
        input_path = os.path.join(tmpdirname, uploaded_file.name)
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        st.write("Processing video...")
        output_path = os.path.join(tmpdirname, 'output.mp4')
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Define a callback function to update the progress bar
        def update_progress(progress):
            progress_bar.progress(progress)
        
        process_video(input_path, output_path, update_progress)

        st.write("Video processed successfully!")

        if os.path.exists(output_path):
            st.video(output_path)
        else:
            st.write("Processed video not found.")