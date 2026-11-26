from ultralytics import YOLO

model = YOLO("runs/detect/runs/baseline_yolov8s/weights/best.pt")
model.info()  # Выведет число слоёв, параметров и вычислительную сложность (GFLOPs)
