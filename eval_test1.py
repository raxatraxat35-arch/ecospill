from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/detect/finetuned_yolov8s-2/weights/best.pt")

    metrics = model.val(
        data="data/oil_spill/data.yaml",
        split="test",
        workers=0,
        device=0,
        name="val_finetuned_test"
    )

    print("\n--- Итоговые метрики на контрольном Test Set (53 фото) ---")
    print(f"mAP@50:    {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")

if __name__ == "__main__":
    main()