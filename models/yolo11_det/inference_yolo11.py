#!/usr/bin/env python3
import argparse
from ultralytics import YOLO
"""
Run inference with a trained YOLOv11 detector.
  # Predict on all images in the val split
  python infer_yolo11_ecp.py \
    --weights runs/detect/train_ecp_yolo11/weights/best.pt \
    --source datasets/ecp-yolo/images/val \
    --imgsz 1280 --conf 0.25 --device 0

  # Predict on a single image
  python infer_yolo11_ecp.py \
    --weights runs/detect/train_ecp_yolo11/weights/best.pt \
    --source /path/to/image.png
"""



def parse_args():
    ap = argparse.ArgumentParser("YOLOv11 inference")
    ap.add_argument("--weights", type=str, default="/home/672511/contrastive-sdxl/models/yolo11_det/runs/detect/train_ecp_yolo11/weights/best.pt",
                     help="Path to trained weights .pt")
    ap.add_argument("--source", type=str, required=True,
                    help="Image path, directory")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="0")  # 'cpu' or '0', '0,1'
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--save", type=lambda x: str(x).lower() == "true", default=True,
                    help="Save visualized predictions")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="predict_ecp_yolo11")
    ap.add_argument("--exist_ok", action="store_true", help="Overwrite if project/name exists")
    return ap.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    preds = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        save=args.save,            # save annotated images/video
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    for i, r in enumerate(preds):
        # r.boxes contains xyxy, conf, cls per detection
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"[{i}] {r.path} -> {n} detections, saved={getattr(r, 'save_dir', None)}")


if __name__ == "__main__":
    main()