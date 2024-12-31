from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train3/weights/best.pt")

# Define class names based on your dataset
class_names = [
    "fn_ar",  # 0
    "ln_ar",  # 1
    "fn_fr",  # 2
    "ln_fr",  # 3
    "bd_fr",  # 4
    "bp_ar",  # 5
    "bp_fr",  # 6
    "vd_fr",  # 7
    "cin_fr"  # 8
]

# Path to the test image
image_path = "rajl.jpeg"  # Replace with the path to your test image

# Run inference on the test image
results = model.predict(source=image_path, save=False, show=False)  # save=False: Do not save automatically

# Open the image using OpenCV
image = cv2.imread(image_path)

# Get the detection results
for result in results:
    boxes = result.boxes  # Detected bounding boxes
    
    # Prepare data for NMS
    nms_boxes = []
    confidences = []
    class_ids = []
    
    for box in boxes:
        # Extract bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers
        
        # Extract confidence score and class label
        conf = float(box.conf[0])  # Confidence score
        cls = int(box.cls[0])  # Class ID
        
        # Store information for NMS
        nms_boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, w, h) format
        confidences.append(conf)
        class_ids.append(cls)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    
    # Draw the selected boxes
    if len(indices) > 0:
        for i in indices.flatten():  # Extract indices returned by NMS
            x, y, w, h = nms_boxes[i]  # Extract box (x, y, w, h)
            conf = confidences[i]  # Extract confidence
            cls = class_ids[i]  # Extract class ID
            
            # Map class ID to class name
            label = f"{class_names[cls]} ({conf:.2f})"
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)  # Green box with thickness 2
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Display the image with detections
cv2.imshow("Detected Fields", image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
