import argparse
import csv
import os

import cv2
from ultralytics import YOLO

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def list_sources(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(os.listdir(path))
    return [os.path.join(path, f) for f in files if f.lower().endswith(IMAGE_EXT + VIDEO_EXT)]


def main():
    parser = argparse.ArgumentParser(description="False-positive test on spill-free frames (T8)")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="clean video, image folder, or folder of clean videos")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--stride", type=int, default=15, help="test every Nth video frame")
    parser.add_argument("--limit", type=float, default=0.20, help="pass if FP rate is at or below this")
    parser.add_argument("--out", default="fp_test_results")
    args = parser.parse_args()

    model = YOLO(args.weights)
    os.makedirs(os.path.join(args.out, "flagged"), exist_ok=True)
    log_path = os.path.join(args.out, "fp_log.csv")

    total = flagged = 0
    with open(log_path, "w", newline="") as log_file:
        log = csv.writer(log_file)
        log.writerow(["source", "frame", "num_boxes", "max_conf"])

        for src in list_sources(args.source):
            is_video = src.lower().endswith(VIDEO_EXT)
            results = model.predict(
                source=src,
                conf=args.conf,
                vid_stride=args.stride if is_video else 1,
                device=0,
                stream=True,
                verbose=False,
            )
            for i, r in enumerate(results):
                total += 1
                n = len(r.boxes)
                max_conf = float(r.boxes.conf.max()) if n else 0.0
                log.writerow([os.path.basename(src), i, n, round(max_conf, 3)])
                if n:
                    flagged += 1
                    name = f"{os.path.splitext(os.path.basename(src))[0]}_{i:05d}.jpg"
                    cv2.imwrite(os.path.join(args.out, "flagged", name), r.plot())

    if total == 0:
        print("No frames found in", args.source)
        return

    rate = flagged / total
    verdict = "PASS" if rate <= args.limit else "FAIL"
    print(f"\nWeights:        {args.weights}")
    print(f"Threshold:      conf = {args.conf}")
    print(f"Frames tested:  {total}")
    print(f"Frames with a false alarm: {flagged}")
    print(f"False-positive rate: {rate:.1%}  (limit {args.limit:.0%})  -> {verdict}")
    print(f"Per-frame log:  {log_path}")
    print(f"Flagged frames: {os.path.join(args.out, 'flagged')}")


if __name__ == "__main__":
    main()
