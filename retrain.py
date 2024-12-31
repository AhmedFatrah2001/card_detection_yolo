from ultralytics import YOLO

# Load the existing trained model
model = YOLO("runs/detect/train2/weights/best.pt")  # Path to your previously trained weights

# Train on the new dataset
results = model.train(
    data="./dataset2/data.yaml",  # Path to new dataset YAML
    epochs=50,  # Number of fine-tuning epochs
    imgsz=640,  # Image size
    pretrained=True,  # Use the pretrained weights
    lr0=0.004  # Optional: Lower learning rate for fine-tuning
)

# Save the fine-tuned model
model.save("runs/detect/train3/weights/best.pt")
