import glob
import os
import cv2

# Точные имена ваших видеофайлов в корне проекта C:\ecospill\
video_names = ["clean_water", "clean_water1"]
frames_per_video = 30  # По 30 кадров с каждого видео = 60 фоновых снимков

img_dir = "data/oil_spill/train/images"
lbl_dir = "data/oil_spill/train/labels"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

total_saved = 0

for idx, base_name in enumerate(video_names, start=1):
    matches = glob.glob(f"{base_name}.*")
    if not matches:
        print(f"Файл '{base_name}' не найден в папке C:\\ecospill\\")
        continue

    video_path = matches[0]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frames_per_video)

    saved_from_this = 0
    frame_idx = 0

    while cap.isOpened() and saved_from_this < frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            resized = cv2.resize(frame, (640, 640))
            out_name = f"bg_clean_{idx}_{saved_from_this:03d}"

            cv2.imwrite(os.path.join(img_dir, f"{out_name}.jpg"), resized)
            open(os.path.join(lbl_dir, f"{out_name}.txt"), "w").close()

            saved_from_this += 1
            total_saved += 1

        frame_idx += 1

    cap.release()
    print(f"Обработано '{video_path}': извлечено {saved_from_this} кадров.")

print(f"\nГотово! Всего добавлено в train: {total_saved} фото чистой воды.")