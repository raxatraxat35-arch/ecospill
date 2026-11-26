from ultralytics import YOLO

def main():
    # Ultralytics 8.4.212 автоматически скачает базовые веса yolo26s.pt
    model = YOLO("yolo26s.pt")

    model.train(
        data="data/oil_spill/data.yaml",
        epochs=100,
        patience=20,
        imgsz=640,
        batch=8,
        workers=0,
        cache=False,
        device=0,
        project="runs/detect",
        name="baseline_yolo26s"
    )

if __name__ == "__main__":
    main()