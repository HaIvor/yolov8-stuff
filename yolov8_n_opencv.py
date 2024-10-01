import random
import cv2
import numpy as np
from ultralytics import YOLO
import os
import subprocess
import torch

# Define a function to get an incrementing filename like test1.mp4, test2.mp4, etc.
def get_incrementing_filename(base_filename, extension):
    counter = 1
    filename = f"{base_filename}{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}{counter}{extension}"
    return filename

# Define the output filename before using it
output_filename = "output_with_detections.mp4"
print("Saving video to:", os.path.abspath(output_filename))

# Load class names
with open("utils/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for each class
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in class_list
]

# Load YOLOv8 model (no need for "v8" argument)
model = YOLO("weights/yolov8n.pt").to("cuda")
print(f"Device being used in model: {model.device}")
# Open the video file
cap = cv2.VideoCapture("inference/videos/mehmet.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Frame width: {frame_width}, Frame height: {frame_height}, FPS: {fps}")

# Set up the VideoWriter to save the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Or "XVID", "MJPG", etc.
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict using YOLOv8 model
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    if len(detect_params[0].boxes) != 0:
        for box in detect_params[0].boxes:
            # Move tensors to CPU before converting to numpy
            clsID = int(box.cls.cpu().numpy()[0])  # Move to CPU
            conf = box.conf.cpu().numpy()[0]  # Move to CPU
            bb = box.xyxy.cpu().numpy()[0]  # Move to CPU

            # Draw the detection box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )

            # Display class name and confidence with 2 decimal places
            label = f"{class_list[clsID]} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    # Save the frame to the output video
    out.write(frame)

    # Display the frame in a window
    cv2.imshow("ObjectDetection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

# Get an incrementing filename like test1.mp4, test2.mp4, etc.
output_reencoded_base = "output_vid"
output_reencoded = get_incrementing_filename(output_reencoded_base, ".mp4")

# Automate the FFmpeg command to re-encode the video and preserve the audio
ffmpeg_command = [
    'ffmpeg',
    '-y',                # Automatically overwrite the output file
    '-i', output_filename,  # Input the generated video with YOLO detections
    '-i', 'inference/videos/mehmet.mp4',  # Input the original video (to copy audio)
    '-c:v', 'libx264',   # Re-encode the video to H.264
    '-c:a', 'aac',       # Re-encode the audio to AAC
    '-b:a', '192k',      # Set audio bitrate to 192kbps
    '-map', '0:v:0',     # Use the video stream from the YOLO output (first input)
    '-map', '1:a:0',     # Use the audio stream from the original video (second input)
    output_reencoded     # Output filename
]

try:
    print(f"Re-encoding video as {output_reencoded} with FFmpeg...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"Video re-encoded successfully: {output_reencoded}")
    
    # Delete the intermediate video without audio
    if os.path.exists(output_filename):
        os.remove(output_filename)
        print(f"Deleted the temporary file: {output_filename}")
        
except subprocess.CalledProcessError as e:
    print(f"Error during re-encoding: {e}")
