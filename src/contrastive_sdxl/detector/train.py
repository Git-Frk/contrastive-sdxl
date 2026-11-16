import argparse
import json
import os
from pathlib import Path

from ..arguments import positive_int, str_bool


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train YOLOv11 on ECP and validate the best checkpoint.")
    parser.add_argument("--data", default="data/yolo/ecp/ecp.yaml")
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--imgsz", type=positive_int, default=1280)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", type=str_bool, nargs="?", const=True, default=False)
    parser.add_argument("--mosaic", type=float, default=1)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--name", default="yolo11")
    parser.add_argument("--resume", action="store_true", help="Resume the checkpoint supplied by --model.")
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name.")
    return args


def train_detector(args):
    output_dir = args.output_dir.resolve()
    for name in ("checkpoints", "generated", "evaluations", "logs"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(output_dir / "logs")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output_dir / "logs" / "ultralytics"))
    from ultralytics import YOLO
    from ultralytics.data import utils as data_utils
    from ultralytics.utils import checks
    from .trainer import DetectorTrainer

    # Dataset preparation does not need to download a font.
    checks.check_font = data_utils.check_font = lambda *a, **kw: Path("Arial.ttf")
    checkpoints_dir = output_dir / "checkpoints" / args.name
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose another --name or resume a checkpoint: {checkpoints_dir}")
    if args.resume and not Path(args.model).is_file():
        raise FileNotFoundError("--resume requires --model to name an existing checkpoint.")
    model_path = Path(args.model)
    if not model_path.exists() and model_path.name == args.model and model_path.suffix == ".pt":
        model_path = output_dir / "checkpoints" / "pretrained" / args.model
        model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    model.train(trainer=DetectorTrainer, data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, amp=args.amp, mosaic=args.mosaic,
                hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, lr0=args.lr0, lrf=args.lrf,
                seed=args.seed, project=str(output_dir / "logs"), name=args.name,
                resume=str(Path(args.model).resolve()) if args.resume else False,
                save_period=0, verbose=True, plots=False)
    metrics = model.val(data=args.data, device=args.device, save_json=args.save_json,
                        project=str(output_dir / "evaluations"), name=args.name, plots=False)
    summary = {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}
    evaluations_dir = Path(metrics.save_dir)
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    (evaluations_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return metrics


def main(argv=None):
    train_detector(parse_args(argv))


if __name__ == "__main__":
    main()
