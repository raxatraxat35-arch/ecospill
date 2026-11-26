from ultralytics import YOLO

# Загружаем ваши обученные веса
model = YOLO("runs/detect/runs/baseline_yolov8s/weights/best.pt")

# Обрабатываем видео на GPU:
# save=True — сохранить итоговый ролик с нарисованными рамками
# show=True — открывать окно предпросмотра во время обработки
results = model.predict(
    source="test_video2.mp4",
    conf=0.45,
    device=0,
    save=True,
    show=True,
    stream=True
)

# Прогоняем видео кадр за кадром (защищает оперативную память на длинных роликах)
for _ in results:
    pass