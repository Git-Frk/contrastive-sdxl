#!/usr/bin/env python3
import argparse, json, random, shutil
from collections import defaultdict
import os
from pathlib import Path
from ultralytics.data.converter import convert_coco

# ---------- helpers ----------
def read_json(p: Path):
    return json.loads(p.read_text())

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))

def split_ids_by_ratio(ids, ratios, seed=0):
    random.Random(seed).shuffle(ids)
    n = len(ids)  # n --> total ids
    t = int(ratios[0] * n)  # t --> train ids count
    v = int(ratios[1] * n)  # n --> val ids count
    train = ids[:t]
    val = ids[t:t+v]
    test = ids[t+v:] if len(ratios) == 3 else []
    return train, val, test


def build_subset_coco(full_annotations, keep_image_ids, drop_crowd=True, keep_categories=None):

    idset = set(keep_image_ids)

    images = []
    for img in full_annotations["images"]:
        if img["id"] not in idset:
            continue
        img_name = dict(img)  # shallow copy
        orig_path = img_name.get("file_name", "")
        img_name["orig_file_name"] = orig_path  # keep full relative path 
        img_name["file_name"] = Path(orig_path).name  # basename 
        images.append(img_name)

    # Filter annotations
    anno = []
    for a in full_annotations.get("annotations", []):
        if a["image_id"] in idset:
            if drop_crowd and a.get("iscrowd", False):
                continue
            anno.append(a)

    # Filter categories (optional)
    cats = full_annotations.get("categories", [])
    if keep_categories:
        names = set(keep_categories)
        cats = [c for c in cats if c.get("name") in names]
        keep_cat_ids = {c["id"] for c in cats}
        anno = [a for a in anno if a["category_id"] in keep_cat_ids]

    return {
        "info": full_annotations.get("info", {}),
        "licenses": full_annotations.get("licenses", []),
        "categories": cats,
        "images": images,
        "annotations": anno,
    }

def main():

    coco_json = Path("/home/672511/Pedestron/datasets/EuroCity/day_train_all.json")
    images_root = Path("/home/672511/Pedestron/datasets/EuroCity/ECP/day/img/train")

    ap = argparse.ArgumentParser("ECP hybrid splitter + YOLO converter")
    ap.add_argument("--coco_json", default=coco_json, type=Path,
                    help="Corrected COCO file with images+annotations (e.g., /mnt/data/train.json)")
    ap.add_argument("--images_root", default=images_root, type=Path,
                    help="Filesystem root that prefixes the COCO file_name paths (folder that contains ECP/...)")
    ap.add_argument("--out_root", default="datasets/ecp-yolo", type=Path,
                    help="Output root for converted dataset")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.0,
                    help="Set >0 to create a test split (e.g., 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-crowd", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--copy-images", type=lambda x: str(x).lower() == "true", default=True,
                    help="true=copy, false=symlink (default)")
    ap.add_argument("--names", nargs="*", default=["pedestrian"],
                    help="names for YAML; leave as your target classes")
    args = ap.parse_args()

    full_annotations = read_json(args.coco_json)

    # --- gather image ids and random split on images ---
    all_ids = [img["id"] for img in full_annotations["images"]]
    train_ids, val_ids, test_ids = split_ids_by_ratio(all_ids,
        (1 - args.val_ratio - args.test_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    # --- build train, val, test COCO JSONs  ---
    splits = ["train", "val"] + (["test"] if args.test_ratio > 0 else [])
    splits_json_dir = args.out_root.parent / "coco_splits"
    splits_json_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        sub_split = build_subset_coco(full_annotations, split_ids[split], drop_crowd=args.drop_crowd)
        write_json(splits_json_dir / f"{split}.json", sub_split)

    # --- run Ultralytics converter per split ---
    # We call convert_coco on the folder (it converts all JSONs inside).
    # Ultralytics writes split-specific images/labels named after each JSON stem.
    # We'll then standardize the layout to images/{split} & labels/{split}.
    convert_coco(
        labels_dir=str(splits_json_dir),
        save_dir=str(args.out_root),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

    # Standardize layout (rename the image/label subfolders created by Ultralytics)
    # We expect folders like images/train and labels/train, because stems are train/val/test.
    # If Ultralytics uses exact stems already, this can be a no-op.
    # Ensure images exist under out_root/images/<split>
    (args.out_root / "images").mkdir(exist_ok=True)
    (args.out_root / "labels").mkdir(exist_ok=True)

    for split in splits:
        # Find any images/<stem> and move/merge to images/<split>
        src_img = args.out_root / "images" / split
        src_lbl = args.out_root / "labels" / split
        src_img.mkdir(parents=True, exist_ok=True)
        src_lbl.mkdir(parents=True, exist_ok=True)

        # nothing to do if already in place
        # (Ultralytics uses the json stem as subfolder name.)

    # --- write YAML ---
    yaml_lines = [
        f"path: {args.out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if args.test_ratio > 0:
        yaml_lines.append("test: images/test")
    yaml_lines.append(f"names: [{', '.join(args.names)}]")
    (args.out_root / "ecp.yaml").write_text("\n".join(yaml_lines))

    # --- copy images into split folders  ---
    # Ultralytics' converter writes label txts and an images.txt listing.
    # To ensure images are accessible under images/<split>, we copy from images_root.
    def link_copy_images(split_name):
        list_file = args.out_root / "images" / f"{split_name}.txt"
        if not list_file.exists():
            # Ultralytics sometimes writes images/{split}/images.txt or none depending on version.
            # If missing, we reconstruct from the split json.
            src_json = splits_json_dir / f"{split_name}.json"
            data = read_json(src_json)
            base_dir = "/home/672511/Pedestron/datasets/EuroCity"
            paths = [img["orig_file_name"] for img in data["images"]]
            paths = [os.path.join(base_dir, path) for path in paths]
        else:
            paths = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
            paths = [os.path.join(base_dir, path) for path in paths]

        out_dir = args.out_root / "images" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            src = (args.images_root / p).resolve()
            dst = out_dir / Path(p).name
            if not dst.exists():
                try:
                    if args.copy_images:
                        shutil.copy2(src, dst)
                    else:
                        dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)

    for split in splits:
        link_copy_images(split)

    # quick report
    print("Done.")
    for s in splits:
        print(f"{s}: {len(split_ids[s])} images")
    print(f"YAML: {(args.out_root / 'ecp.yaml').as_posix()}")

if __name__ == "__main__":
    main()
