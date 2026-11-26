from ultralytics import YOLO

def main():
    # Загружаем предобученную легкую архитектуру YOLOv8s
    model = YOLO("yolov8s.pt")

    # Запускаем обучение с параметрами под 4 ГБ VRAM и Windows
    model.train(
        data="data/oil_spill/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,        # Безопасный размер батча для RTX 3050 Ti
        workers=2,      # Оптимально для стабильной работы на Windows
        device=0,       # Обучение на GPU
        project="runs",
        name="baseline_yolov8s"
    )

if __name__ == "__main__":
    main()