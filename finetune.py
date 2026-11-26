from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/runs/baseline_yolov8s/weights/best.pt")

    model.train(
        data="data/oil_spill/data.yaml",
        epochs=40,
        imgsz=640,
        batch=8,
        lr0=0.001,
        workers=0,               # Отключаем мультипроцессинг DataLoader (решает WinError 1455)
        cache=False,             # Не забиваем оперативную память кэшем картинок
        device=0,
        project="runs/detect",
        name="finetuned_yolov8s"
    )

if __name__ == "__main__":
    main()