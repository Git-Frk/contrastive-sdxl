from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer


class DetectorTrainer(DetectionTrainer):
    """Keep detector weights separate from logs, including resumed runs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_dir = Path(self.args.project).parent
        self.wdir = output_dir / "checkpoints" / self.args.name
        if self.wdir.is_symlink() or any((self.wdir / name).is_symlink() for name in ("best.pt", "last.pt")):
            raise FileExistsError(f"Choose another --name to keep linked checkpoints unchanged: {self.wdir}")
        if self.wdir.exists() and any(self.wdir.iterdir()) and not self.resume:
            raise FileExistsError(f"Choose another --name or resume a checkpoint: {self.wdir}")
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last = self.wdir / "last.pt"
        self.best = self.wdir / "best.pt"

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            # Ultralytics restores old output paths from the checkpoint by default.
            self.args.project = overrides["project"]
            self.args.name = overrides["name"]
            self.args.save_dir = None
            self.args.data = overrides["data"]
