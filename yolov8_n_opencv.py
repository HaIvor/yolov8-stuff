import random
import cv2
import numpy as np
from ultralytics import YOLO
import os
import subprocess

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

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt", "v8")

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
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            box = detect_params[0].boxes[i]
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw the detection box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )

            # Display class name and confidence
            label = f"{class_list[clsID]} {round(conf, 3)}"
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

# Automate the FFmpeg command to re-encode the video and preserve the audio

output_reencoded = "output_fixed_with_audio.mp4"
ffmpeg_command = [
    'ffmpeg',
    '-i', output_filename,  # Input the generated video with YOLO detections
    '-c:v', 'libx264',      # Re-encode the video to H.264
    '-c:a', 'copy',         # Preserve the original audio
    output_reencoded        # Output filename
]

try:
    print("Re-encoding video with FFmpeg...")
    subprocess.run(ffmpeg_command, check=True)
    print(f"Video re-encoded successfully: {output_reencoded}")
except subprocess.CalledProcessError as e:
    print(f"Error during re-encoding: {e}")
