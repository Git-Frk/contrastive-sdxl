#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import checks as uchecks
import ultralytics.data.utils as dutils

def _no_font(*args, **kwargs):
    return Path("Arial.ttf")

uchecks.check_font = _no_font
dutils.check_font  = _no_font


def parse_args():
    ap = argparse.ArgumentParser("YOLOv11 training on ECP")
    ap.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    ap.add_argument("--model", type=str, default="yolo11l.pt",
                    help="Base model or checkpoint path (e.g., yolo11l.pt or runs/detect/train/weights/last.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-batch")
    ap.add_argument("--device", type=str, default="0", help="GPU id(s), e.g. '0' or '0,1' or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.7)
    ap.add_argument("--hsv_v", type=float, default=0.4)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial LR; leave None for default")
    ap.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction; leave None for default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="train_ecp_yolo11")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint in project/name")
    ap.add_argument("--save_json", action="store_true", help="Save COCO-format JSON results on val")
    return ap.parse_args()


def main():
    args = parse_args()

    # Initialize model (fresh or from checkpoint/base weights)
    model = YOLO(args.model)

    # Build training kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=args.amp,
        mosaic=args.mosaic,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        save_period=0,   # set >0 to save every N epochs
        verbose=True,
        plots=False 
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf

    # Train
    results = model.train(**train_kwargs)

    # Validate on val split (and optionally save COCO JSON metrics)
    val_kwargs = dict(
        data=args.data,
        device=args.device,
        save_json=args.save_json
    )
    metrics = model.val(**val_kwargs)

    # Print key metrics and best weights path
    print("\n=== Training done ===")
    print(f"mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()