from ultralytics import YOLO

def main():
    # Точный путь из вашего лога обучения
    model = YOLO("runs/detect/runs/detect/baseline_yolo26s/weights/best.pt")

    metrics = model.val(
        data="data/oil_spill/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        device=0
    )

    print("\n--- Итоговые метрики YOLO26s на TEST ---")
    print(f"mAP@50:    {metrics.box.map50 * 100:.1f}%")
    print(f"mAP@50-95: {metrics.box.map * 100:.1f}%")
    print(f"Precision: {metrics.box.mp * 100:.1f}%")
    print(f"Recall:    {metrics.box.mr * 100:.1f}%")

if __name__ == "__main__":
    main()