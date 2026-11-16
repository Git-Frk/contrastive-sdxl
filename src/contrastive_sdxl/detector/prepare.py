import argparse
import json
import random
import shutil
from pathlib import Path

from ..arguments import str_bool


def split_ids_by_ratio(ids, ratios, seed=42):
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    t, v = int(ratios[0] * len(ids)), int(ratios[1] * len(ids))
    return ids[:t], ids[t:t + v], ids[t + v:]


def build_subset_coco(full_annotations, keep_image_ids, drop_crowd=True, keep_categories=None):
    idset = set(keep_image_ids)
    cats = full_annotations.get("categories", [])
    if keep_categories:
        cats = [cat for cat in cats if cat["name"] in keep_categories]
    cat_ids = {cat["id"] for cat in cats}
    images = []
    for img in full_annotations["images"]:
        if img["id"] in idset:
            img = dict(img)
            img["orig_file_name"] = img.get("orig_file_name", img["file_name"])
            img["file_name"] = Path(img["file_name"]).name
            images.append(img)
    anno = [a for a in full_annotations.get("annotations", [])
            if a["image_id"] in idset and a["category_id"] in cat_ids
            and not (drop_crowd and a.get("iscrowd", False))]
    return {"info": full_annotations.get("info", {}), "licenses": full_annotations.get("licenses", []),
            "images": images, "annotations": anno, "categories": cats}


def coco_labels(data):
    """Convert COCO xywh boxes to normalized YOLO rows, including empty images."""
    categories = sorted(data["categories"], key=lambda cat: cat["id"])
    class_ids = {cat["id"]: i for i, cat in enumerate(categories)}
    images = {img["id"]: img for img in data["images"]}
    labels = {key: [] for key in images}
    stems = [Path(img["file_name"]).stem for img in images.values()]
    if len(stems) != len(set(stems)):
        raise ValueError("Image basenames must be unique when creating flat YOLO labels.")
    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        if img["width"] <= 0 or img["height"] <= 0:
            raise ValueError(f"Invalid dimensions for {img['file_name']}")
        row = [class_ids[ann["category_id"]], (x + w / 2) / img["width"],
               (y + h / 2) / img["height"], w / img["width"], h / img["height"]]
        if row not in labels[ann["image_id"]]:
            labels[ann["image_id"]].append(row)
    return labels, [cat["name"] for cat in categories]


def write_labels(data, labels_dir):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels, names = coco_labels(data)
    for img in data["images"]:
        path = labels_dir / (Path(img["file_name"]).stem + ".txt")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Label already exists: {path}")
        rows = labels[img["id"]]
        path.write_text("".join(" ".join(f"{value:g}" for value in row) + "\n" for row in rows))
    return names


def prepare(args):
    full_annotations = json.loads(args.coco_json.read_text())
    ids = [img["id"] for img in full_annotations["images"]]
    data = build_subset_coco(full_annotations, ids, args.drop_crowd, args.names)
    if not data["images"] or not data["categories"]:
        raise ValueError("No images or selected categories in the annotations.")
    if args.mode == "labels":
        if args.labels_dir.exists() and any(args.labels_dir.iterdir()):
            raise FileExistsError(f"Choose an empty labels directory: {args.labels_dir}")
        write_labels(data, args.labels_dir)
        return args.labels_dir
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"Choose an empty detector dataset directory: {args.out_root}")
    sources = {}
    for img in data["images"]:
        original = Path(img["orig_file_name"])
        candidates = (args.images_root / original, args.images_root / img["file_name"])
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Cannot find {original} beneath {args.images_root}")
        sources[img["id"]] = source.resolve()
    ratios = (1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio)
    train_ids, val_ids, test_ids = split_ids_by_ratio(ids, ratios, args.seed)
    split_ids = {"train": train_ids, "val": val_ids}
    if args.test_ratio > 0:
        split_ids["test"] = test_ids
    for split, keep_ids in split_ids.items():
        subset = build_subset_coco(data, keep_ids, args.drop_crowd, args.names)
        names = write_labels(subset, args.out_root / "labels" / split)
        images_dir = args.out_root / "images" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in subset["images"]:
            dst = images_dir / img["file_name"]
            if args.copy_images:
                shutil.copy2(sources[img["id"]], dst)
            else:
                dst.symlink_to(sources[img["id"]])
        annotations_dir = args.out_root / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        (annotations_dir / f"{split}.json").write_text(json.dumps(subset))
    import yaml
    config = {"path": str(args.out_root.resolve()), "train": "images/train", "val": "images/val", "names": names}
    if args.test_ratio > 0:
        config["test"] = "images/test"
    path = args.out_root / "ecp.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare an ECP detector dataset or SDXL guidance labels.")
    parser.add_argument("--mode", choices=("train", "labels"), required=True)
    parser.add_argument("--coco_json", type=Path, default=Path("data/ECP/annotations/day_train_all.json"))
    parser.add_argument("--images_root", type=Path, default=Path("data/ECP/train_src"))
    parser.add_argument("--out_root", type=Path, default=Path("data/yolo/ecp"))
    parser.add_argument("--labels_dir", type=Path, default=Path("data/ECP/labels/train_src"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-crowd", type=str_bool, nargs="?", const=True, default=True)
    parser.add_argument("--copy-images", type=str_bool, nargs="?", const=True, default=True)
    parser.add_argument("--names", nargs="+", default=["pedestrian"])
    args = parser.parse_args(argv)
    if not (0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        parser.error("Split ratios must be nonnegative and sum to less than one.")
    return args


def main(argv=None):
    path = prepare(parse_args(argv))
    print(f"Prepared {path}")


if __name__ == "__main__":
    main()
