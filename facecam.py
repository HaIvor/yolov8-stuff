from ultralytics import YOLO
import cv2

# Load YOLOv8 model and set it to run on the GPU (if available)
model = YOLO("weights/yolo11n.pt").to("cuda")

# Confirm which device the model is using (GPU or CPU)
device = model.device
print(f"Model is running on: {device}")

# Start video capture from the default camera (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict using YOLOv8 model on the current frame
    results = model.predict(source=frame)

    # Draw the detection boxes on the frame
    for box in results[0].boxes:
        clsID = int(box.cls.cpu().numpy()[0])  # Get the class ID
        conf = box.conf.cpu().numpy()[0]  # Get the confidence
        bb = box.xyxy.cpu().numpy()[0]  # Get the bounding box coordinates

        # Draw the detection box
        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 2)
        label = f"{model.names[clsID]} {conf:.2f}"
        cv2.putText(frame, label, (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
