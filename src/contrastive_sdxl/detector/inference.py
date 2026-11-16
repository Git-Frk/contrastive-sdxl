import argparse
import os
from pathlib import Path

from ..arguments import positive_int, str_bool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict pedestrians with a trained YOLOv11 detector.")
    parser.add_argument("--weights", default="outputs/checkpoints/yolo11/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=positive_int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_det", type=positive_int, default=300)
    parser.add_argument("--save", type=str_bool, nargs="?", const=True, default=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="yolo11")
    parser.add_argument("--exist_ok", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name.")
    return args


def predict(args):
    output_dir = args.output_dir.resolve()
    for name in ("checkpoints", "generated", "evaluations", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / "logs" / "ultralytics"))
    from ultralytics import YOLO

    model = YOLO(args.weights)
    count = 0
    for result in model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                device=args.device, max_det=args.max_det, save=args.save,
                                project=str(output_dir / "generated"), name=args.name,
                                exist_ok=args.exist_ok, stream=True):
        detections = 0 if result.boxes is None else len(result.boxes)
        print(f"{result.path}: {detections} detections")
        count += 1
    return count


def main(argv=None):
    predict(parse_args(argv))


if __name__ == "__main__":
    main()
