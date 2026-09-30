import json
from pathlib import Path
import torch
from ultralytics.data.converter import convert_coco

# ---------- helpers ----------
def read_json(p: Path):
    return json.loads(p.read_text())

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))

def build_subset_coco(full_annotations, keep_image_ids, drop_crowd=True, keep_categories=None):
    idset = set(keep_image_ids)

    # images: keep and normalize file_name to basename, store original in orig_file_name
    images = []
    for img in full_annotations["images"]:
        if img["id"] not in idset:
            continue
        im = dict(img)  # shallow copy
        orig_path = im.get("file_name", "")
        im["orig_file_name"] = orig_path
        im["file_name"] = Path(orig_path).name  # basename for flat label stems
        images.append(im)

    # annotations: keep by image_id, drop crowd if requested, optional class filter
    anno = []
    for a in full_annotations.get("annotations", []):
        if a["image_id"] in idset:
            if drop_crowd and a.get("iscrowd", False):
                continue
            anno.append(a)

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

# ---------- letterbox math (scale + symmetric pad) ----------
def letterbox_gain_pad(orig_hw, dst_hw=(512, 512)):
    H0, W0 = orig_hw
    H, W = dst_hw
    r = min(W / W0, H / H0)                 # uniform scale (keep aspect)
    newW, newH = int(W0 * r), int(H0 * r)
    padW, padH = (W - newW) / 2, (H - newH) / 2
    return r, padW, padH

def transform_cxcywh_norm_to_lb512(cxcywh_norm, orig_hw, dst_hw=(512, 512)):
    """
    cxcywh_norm: (N,4) tensor/list with normalized [cx,cy,w,h] in ORIGINAL frame (W0,H0).
    Returns normalized [cx,cy,w,h] in the 512x512 LETTERBOXED frame.
    """
    import torch
    t = torch.as_tensor(cxcywh_norm, dtype=torch.float32)
    if t.numel() == 0:
        return t  # empty

    H0, W0 = orig_hw
    H, W = dst_hw
    r, padW, padH = letterbox_gain_pad(orig_hw, dst_hw)

    # unnormalize to original absolute pixels
    cx_abs = t[:, 0] * W0
    cy_abs = t[:, 1] * H0
    w_abs  = t[:, 2] * W0
    h_abs  = t[:, 3] * H0

    # convert to corners
    x1 = cx_abs - 0.5 * w_abs
    y1 = cy_abs - 0.5 * h_abs
    x2 = cx_abs + 0.5 * w_abs
    y2 = cy_abs + 0.5 * h_abs

    # scale+pad into dst frame
    x1 = x1 * r + padW;  y1 = y1 * r + padH
    x2 = x2 * r + padW;  y2 = y2 * r + padH

    # back to cxcywh and normalize by dst size
    cx = ((x1 + x2) * 0.5) / W
    cy = ((y1 + y2) * 0.5) / H
    ww = (x2 - x1) / W
    hh = (y2 - y1) / H

    # (optional) clip to [0,1] to be safe
    cx = cx.clamp(0.0, 1.0)
    cy = cy.clamp(0.0, 1.0)
    ww = ww.clamp(0.0, 1.0)
    hh = hh.clamp(0.0, 1.0)

    return torch.stack([cx, cy, ww, hh], dim=-1)

# ---------- main conversion ----------
def convert_annotations():
    # --- config ---
    coco_json = Path("/home/672511/Pedestron/datasets/EuroCity/day_train_all.json")
    out_root = Path("datasets/loss-labels")
    drop_crowd = True
    target_names = ["pedestrian"]  # filter to pedestrians

    # --- read and subset (no image filtering; we keep ALL images) ---
    full_annotations = read_json(coco_json)
    all_ids = [img["id"] for img in full_annotations["images"]]

    json_filename = "annotations"
    json_dir = out_root.parent / "coco_anns"
    json_dir.mkdir(parents=True, exist_ok=True)

    sub_split = build_subset_coco(
        full_annotations,
        all_ids,
        drop_crowd=drop_crowd,
        keep_categories=target_names
    )
    split_json_path = json_dir / f"{json_filename}.json"
    write_json(split_json_path, sub_split)

    # --- run Ultralytics converter (writes labels/<split>/*.txt for images that still have boxes) ---
    convert_coco(
        labels_dir=str(json_dir),
        save_dir=str(out_root),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
    )

    # --- ensure one label per image (empty .txt for backgrounds) ---
    labels_dir = out_root / "labels" / json_filename               # original YOLO labels (orig frame)
    labels_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(split_json_path)
    have = {p.stem for p in labels_dir.glob("*.txt")}
    want = {Path(im["file_name"]).stem for im in data["images"]}
    missing = want - have
    print(f"Creating {len(missing)} empty labels...")
    for stem in sorted(missing):
        (labels_dir / f"{stem}.txt").write_text("")

    # --- build a mapping stem -> (H0,W0) from COCO images, then write letterboxed-512 labels ---
    # The COCO 'images' entries include the original width/height that convert_coco used.
    stem_to_hw = {}
    for im in data["images"]:
        stem = Path(im["file_name"]).stem
        # pycococreator puts (width,height) in the image entry
        # keys are typically 'width' and 'height'
        H0 = im.get("height")
        W0 = im.get("width")
        if H0 is None or W0 is None:
            raise ValueError(f"Missing width/height in COCO for {im['file_name']}")
        stem_to_hw[stem] = (H0, W0)

    # output directory for 512×512-letterboxed labels
    labels512_dir = out_root / "labels_lb512" / json_filename
    labels512_dir.mkdir(parents=True, exist_ok=True)

    # iterate all images (includes empty-label ones)
    count_src, count_dst = 0, 0
    for stem in sorted(want):
        count_src += 1
        src = labels_dir / f"{stem}.txt"
        dst = labels512_dir / f"{stem}.txt"
        lines = [ln.strip() for ln in src.read_text().splitlines()] if src.exists() else []

        if not lines:
            dst.write_text("")  # keep empty file if no objects
            continue

        # parse rows [cls cx cy w h] (normalized to original)
        rows = []
        for ln in lines:
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls, cx, cy, w, h = map(float, parts[:5])
            rows.append([cls, cx, cy, w, h])

        if not rows:
            dst.write_text("")
            continue

        t = torch.tensor(rows, dtype=torch.float32)  # (N,5)
        cls_col = t[:, 0:1]
        cxcywh_norm_orig = t[:, 1:5]                # normalized to (W0,H0)

        H0, W0 = stem_to_hw[stem]
        cxcywh_norm_512 = transform_cxcywh_norm_to_lb512(cxcywh_norm_orig, (H0, W0), (512, 512))

        # reattach class column
        out = torch.cat([cls_col, cxcywh_norm_512], dim=1)

        # write back as YOLO lines
        txt = "\n".join(" ".join(f"{v:.6f}" for v in row.tolist()) for row in out)
        dst.write_text(txt)
        count_dst += 1

    # quick report
    print("Done.")
    print(f"Total images in JSON: {len(data['images'])}")
    print(f"Labels (orig frame) present: {len(have)} (+{len(missing)} empty)")
    print(f"Labels (letterboxed 512) written: {count_dst} (empty preserved for the rest)")
    print(f"Orig labels dir: {labels_dir.resolve()}")
    print(f"LB512 labels dir: {labels512_dir.resolve()}")

def main():
    convert_annotations()

if __name__ == "__main__":
    main()