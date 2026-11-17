"""Build the LEAF LENS crop/OOD train and validation folders from Kaggle ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import stat
import warnings
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from backend.prepare_dataset import OOD_CLASS_NAME, PLANTDOC_CLASS_MAP, TARGET_PREFIXES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
PLANTVILLAGE_CLASSES = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
}
EXPECTED_CROP_CLASSES = PLANTVILLAGE_CLASSES | {"Tomato___Unspecified_leaf"}


def _classify(kind: str, member: str) -> tuple[str, str | None] | None:
    parts = PurePosixPath(member).parts
    if kind == "plantvillage":
        split = next((name for name in ("train", "valid", "val") if name in parts), None)
        if split is None:
            return None
        index = parts.index(split)
        if index + 2 >= len(parts):
            return None
        label = parts[index + 1]
        if label not in PLANTVILLAGE_CLASSES:
            return None
        return label, "train" if split == "train" else "val"
    if kind == "plantdoc":
        if len(parts) < 3 or parts[0] not in {"train", "test"}:
            return None
        label = PLANTDOC_CLASS_MAP.get(parts[1])
        if label is None:
            return None
        return label, "train" if parts[0] == "train" else "val"
    if kind == "catdog":
        if "training_set" in parts:
            return OOD_CLASS_NAME, "train"
        if "test_set" in parts:
            return OOD_CLASS_NAME, "val"
        return None
    if kind in {"faces", "coco"}:
        return OOD_CLASS_NAME, None
    raise ValueError(f"Unknown dataset source: {kind}")


def _archives(source_root: Path) -> dict[str, Path]:
    return {
        "plantvillage": source_root / "plantvillage/new-plant-diseases-dataset.zip",
        "plantdoc": source_root / "plantdoc/plantdoc-dataset.zip",
        "catdog": source_root / "cat-and-dog/cat-and-dog.zip",
        "faces": source_root / "human-faces/human-faces-dataset.zip",
        "coco": source_root / "coco128/coco128.zip",
    }


def _validated_digest(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    rejected: Counter[str],
) -> str | None:
    member = PurePosixPath(info.filename)
    if member.is_absolute() or ".." in member.parts:
        rejected["unsafe_path"] += 1
        return None
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK or info.flag_bits & 1:
        rejected["symlink_or_encrypted"] += 1
        return None
    if info.file_size < 1 or info.file_size > MAX_IMAGE_BYTES:
        rejected["size_limit"] += 1
        return None
    if info.file_size / max(info.compress_size, 1) > 200:
        rejected["compression_ratio"] += 1
        return None

    try:
        image_bytes = archive.read(info)
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                rejected["pixel_limit"] += 1
                return None
            image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        rejected["invalid_image"] += 1
        return None
    return hashlib.sha256(image_bytes).hexdigest()


def build_dataset(source_root: Path, dataset_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    dataset_root = dataset_root.resolve()
    if source_root == dataset_root or source_root in dataset_root.parents or dataset_root in source_root.parents:
        raise ValueError("Source and destination roots must not overlap.")

    archives = _archives(source_root)
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Kaggle ZIP archives are missing: {missing}")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset: {dataset_root}")

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    source_order = ("plantvillage", "plantdoc", "catdog", "faces", "coco")

    with ExitStack() as stack:
        opened = {
            kind: stack.enter_context(zipfile.ZipFile(path))
            for kind, path in archives.items()
        }
        for priority, kind in enumerate(source_order):
            for info in opened[kind].infolist():
                if info.is_dir():
                    continue
                suffix = PurePosixPath(info.filename).suffix.lower()
                if suffix not in SUPPORTED_SUFFIXES:
                    continue
                target = _classify(kind, info.filename)
                if target is None:
                    continue
                label, split_hint = target
                digest = _validated_digest(opened[kind], info, rejected)
                if digest is None:
                    continue
                candidates.append(
                    {
                        "kind": kind,
                        "priority": priority,
                        "member": info.filename,
                        "class": label,
                        "split_hint": split_hint,
                        "digest": digest,
                        "suffix": suffix,
                    }
                )
            print(
                f"Validated {kind}: "
                f"{sum(item['kind'] == kind for item in candidates)} images",
                flush=True,
            )

        by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidates:
            by_digest[item["digest"]].append(item)

        unique: list[dict[str, Any]] = []
        conflicting_hashes = 0
        cross_split_duplicates = 0
        for items in by_digest.values():
            if len({item["class"] for item in items}) != 1:
                conflicting_hashes += 1
                continue
            hints = {
                item["split_hint"]
                for item in items
                if item["split_hint"] is not None
            }
            if hints == {"train", "val"}:
                split_hint = "train"
                cross_split_duplicates += 1
            else:
                split_hint = next(iter(hints), None)
            selected = dict(
                min(items, key=lambda item: (item["priority"], item["member"]))
            )
            selected["split_hint"] = split_hint
            unique.append(selected)

        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in unique:
            by_class[item["class"]].append(item)
        classes = set(by_class)
        crop_classes = classes - {OOD_CLASS_NAME}
        if crop_classes != EXPECTED_CROP_CLASSES:
            raise ValueError(
                "The archive set must provide exactly the 18 expected crop classes. "
                f"Missing: {sorted(EXPECTED_CROP_CLASSES - crop_classes)}; "
                f"unexpected: {sorted(crop_classes - EXPECTED_CROP_CLASSES)}."
            )
        if OOD_CLASS_NAME not in classes:
            raise ValueError("No non-leaf images were validated for the OOD class.")
        if not any(name.startswith(TARGET_PREFIXES[0]) for name in crop_classes):
            raise ValueError("Corn classes are missing.")
        if not any(name.startswith(TARGET_PREFIXES[1]) for name in crop_classes):
            raise ValueError("Potato classes are missing.")
        if not any(name.startswith(TARGET_PREFIXES[2]) for name in crop_classes):
            raise ValueError("Tomato classes are missing.")

        rng = random.Random(42)
        assignments: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
        for label, items in sorted(by_class.items()):
            fixed_train = [item for item in items if item["split_hint"] == "train"]
            fixed_val = [item for item in items if item["split_hint"] == "val"]
            unassigned = [item for item in items if item["split_hint"] is None]
            rng.shuffle(unassigned)
            val_needed = max(0, round(len(items) * 0.20) - len(fixed_val))
            val_needed = min(val_needed, len(unassigned))
            assignments["train"].extend(fixed_train)
            assignments["val"].extend(fixed_val)
            assignments["val"].extend(unassigned[:val_needed])
            assignments["train"].extend(unassigned[val_needed:])

        for split, items in assignments.items():
            found = {item["class"] for item in items}
            if found != classes:
                raise ValueError(
                    f"Class folders differ for {split}; missing: "
                    f"{sorted(classes - found)}."
                )

        if dataset_root.exists():
            dataset_root.rmdir()
        for split, items in assignments.items():
            for item in items:
                destination = (
                    dataset_root
                    / split
                    / item["class"]
                    / f"{item['digest'][:24]}{item['suffix']}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                image_bytes = opened[item["kind"]].read(item["member"])
                if hashlib.sha256(image_bytes).hexdigest() != item["digest"]:
                    raise RuntimeError(
                        f"Archive contents changed during extraction: {item['member']}"
                    )
                with destination.open("xb") as output:
                    output.write(image_bytes)
            counts = Counter(item["class"] for item in items)
            print(f"{split}: {len(items)} images across {len(counts)} classes")
            for label, count in sorted(counts.items()):
                print(f"  {count:6d} {label}")

    train_hashes = {item["digest"] for item in assignments["train"]}
    val_hashes = {item["digest"] for item in assignments["val"]}
    if train_hashes & val_hashes:
        raise RuntimeError(
            f"Found {len(train_hashes & val_hashes)} duplicate image(s) across splits."
        )
    print(f"Validated unique images: {len(unique)}")
    print(f"Cross-split duplicate groups assigned to train only: {cross_split_duplicates}")
    print(f"Conflicting-label duplicate groups excluded: {conflicting_hashes}")
    print(f"Rejected invalid/unsafe images: {dict(rejected)}")
    print(f"PlantVillage dataset ready at {dataset_root}")
    return {
        "classes": len(classes),
        "crop_classes": len(crop_classes),
        "train_images": len(assignments["train"]),
        "val_images": len(assignments["val"]),
        "cross_split_duplicates": cross_split_duplicates,
        "conflicting_hashes": conflicting_hashes,
        "rejected": dict(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "DataSet")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "PlantVillage")
    args = parser.parse_args()
    build_dataset(args.source_root, args.dataset_root)


if __name__ == "__main__":
    main()
