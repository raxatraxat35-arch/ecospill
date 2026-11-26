from ultralytics import YOLO

def main():
    # Загружаем лучшие веса после 100 эпох обучения
    model = YOLO("runs/detect/runs/baseline_yolov8s/weights/best.pt")

    # Запускаем проверку на отложенной папке test
    metrics = model.val(data="data/oil_spill/data.yaml", split="test", device=0)

    print("\n--- Результаты на тестовой выборке ---")
    print(f"mAP@50:    {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()