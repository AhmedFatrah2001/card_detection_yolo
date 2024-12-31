from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(data="./yolov8_id_dataset/data.yaml", epochs=50,imgsz=640)