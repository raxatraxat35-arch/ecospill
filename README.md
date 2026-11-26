EcoSpill-Vision: Real-Time Oil Spill Detection Using YOLOv8

Requirements Python 3 and an NVIDIA GPU with CUDA. Install the CUDA build of PyTorch from pytorch.org, then: pip install -r requirements.txt

Data Put your Roboflow API key in downlaod_data.py and run it to download oil-spill-sfzvs version 6 to data/oil_spill. For Stage 2, after Stage 1 is trained, copy 300 images with labels from the Roboflow dataset oil-spill-ujfym into data/oil_spill/train and valid, and put the videos clean_water and clean_water1 in the project root. The video scripts expect test_video.mp4 and clean_video1.mp4 in the project root. Videos are not stored in this repository.

How to run python check_gpu.py python downlaod_data.py python train_baseline.py python eval_test.py python test_model.py python extract_backgrounds.py python finetune.py python test_video.py python test_clean_video.py python train_yolo26s.py python eval_yolo26s.py python test_video_26.py python false_positive.py --weights runs/detect/runs/detect/baseline_yolo26s/weights/best.pt --source clean_video1.mp4 --conf 0.45 --stride 5 python fps_test.py --weights best.engine --source test_video.mp4 --seconds 60 (on the Jetson, after: yolo export model=best.pt format=engine half=True)

Expected results On the test split, eval_test.py gives about 78.5% mAP@50 for Stage 1 and eval_yolo26s.py about 79.8% for YOLO26s. For Stage 2 (about 44.9%), set the weights path in eval_test.py to runs/detect/runs/detect/finetuned_yolov8s-2/weights/best.pt. false_positive.py gives about 3.8% false-alarm frames for YOLO26s on clean_video1.

External resources Ultralytics YOLOv8 and YOLO26 with COCO-pretrained weights, PyTorch, OpenCV and the two Roboflow Universe datasets.