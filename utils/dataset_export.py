"""
The three dataset exporters (YOLO, Pascal VOC, COCO) as plain functions: the pictures to export (`records`, from
`find_workspace_images`), the folders of labels and the classes are passed in, so the desktop window and the web server
use the very same code and nothing here reads the desktop's workspace globals. No tkinter.

`records` = [{'source': path, 'filename': name, 'base': name without extension}] of every dataset folder of the workspace.
`augment` (augmentation.Augmenter) and `preprocess` (export_preprocess.Preprocess) are optional. `rng` shuffles the
pictures before the split (pass random.Random(seed) for a repeatable split).
"""
import copy
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image

from .augment_geometry import coco_annotation, voc_xml_write, yolo_label_text
from .augmentation import write_image


def find_workspace_images(datasets_root, workspace, image_exts):
    """Every picture of the workspace's dataset folders (<workspace>-1, -2 ...), folders and files sorted by name."""
    prefix = workspace + '-'
    records = []
    if not os.path.isdir(datasets_root):
        return records
    folders = sorted(folder for folder in os.listdir(datasets_root)
                     if folder.startswith(prefix) and os.path.isdir(os.path.join(datasets_root, folder)))
    for folder in folders:
        folder_path = os.path.join(datasets_root, folder)
        for filename in sorted(os.listdir(folder_path)):
            if filename.lower().endswith(image_exts):
                records.append({'source': os.path.join(folder_path, filename), 'filename': filename,
                                'base': os.path.splitext(filename)[0]})
    return records


def read_coco_annotations(xml_path, image_width, image_height, categories):
    """Convert Jelibox VOC objects into COCO annotation dictionaries."""
    annotations = []
    if not os.path.exists(xml_path):
        return annotations

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"[COCO] Could not read {xml_path}: {exc}")
        return annotations

    for obj in root.findall('object'):
        name_elem = obj.find('name')
        class_name = (name_elem.text or '').strip() if name_elem is not None else ''
        if not class_name or class_name not in categories:
            continue

        polygon = obj.find('polygon')
        segmentation = []
        points = []
        if polygon is not None:
            point_elements = polygon.findall('point')
            if point_elements:
                for point in point_elements:
                    x_elem = point.find('x')
                    y_elem = point.find('y')
                    if x_elem is not None and y_elem is not None:
                        points.append((float(x_elem.text), float(y_elem.text)))
            else:
                index = 1
                while polygon.find(f'x{index}') is not None and polygon.find(f'y{index}') is not None:
                    points.append((
                        float(polygon.find(f'x{index}').text),
                        float(polygon.find(f'y{index}').text)
                    ))
                    index += 1

        if len(points) >= 3:
            points = [
                (max(0.0, min(float(image_width), x)),
                 max(0.0, min(float(image_height), y)))
                for x, y in points
            ]
            segmentation = [[coord for point in points for coord in point]]
            xmin = min(point[0] for point in points)
            ymin = min(point[1] for point in points)
            xmax = max(point[0] for point in points)
            ymax = max(point[1] for point in points)
            area = 0.5 * abs(sum(
                points[index][0] * points[(index + 1) % len(points)][1] -
                points[(index + 1) % len(points)][0] * points[index][1]
                for index in range(len(points))
            ))
        else:
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            xmin = float(bbox.findtext('xmin', '0'))
            ymin = float(bbox.findtext('ymin', '0'))
            xmax = float(bbox.findtext('xmax', '0'))
            ymax = float(bbox.findtext('ymax', '0'))
            area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)

        xmin = max(0.0, min(float(image_width), xmin))
        ymin = max(0.0, min(float(image_height), ymin))
        xmax = max(xmin, min(float(image_width), xmax))
        ymax = max(ymin, min(float(image_height), ymax))
        annotations.append({
            'category_id': categories[class_name],
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'area': area,
            'segmentation': segmentation,
            'iscrowd': 0
        })

    return annotations


def export_coco(records, voc_dir, workspace, classes, target, train_pct, valid_pct, progress_cb=None, augment=None,
                preprocess=None, rng=random):
    """Export indexed workspace images and VOC annotations as COCO JSON."""
    if not records:
        raise ValueError(f'No images found for workspace {workspace} in datasetsInput')

    rng.shuffle(records)
    total = len(records)
    processed = 0
    train_count = int(round(total * (train_pct / 100.0)))
    valid_count = int(round(total * (valid_pct / 100.0)))
    if train_count + valid_count > total:
        valid_count = max(0, total - train_count)

    splits = {
        'train': records[:train_count],
        'val': records[train_count:train_count + valid_count],
        'test': records[train_count + valid_count:]
    }
    categories = {name: index + 1 for index, name in enumerate(classes)}
    xml_root = voc_dir

    for split_name, split_records in splits.items():
        image_dir = os.path.join(target, 'images', split_name)
        os.makedirs(image_dir, exist_ok=True)
        coco = {
            'info': {'description': f'Jelibox {workspace} dataset'},
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': [
                {'id': category_id, 'name': name, 'supercategory': 'object'}
                for name, category_id in categories.items()
            ]
        }
        annotation_id = 1
        image_id = 0

        for record in split_records:
            image_id += 1
            try:
                with Image.open(record['source']) as image:
                    image_width, image_height = image.size
                output_name = record['filename']
                destination = os.path.join(image_dir, output_name)
                if os.path.exists(destination):
                    output_name = f"{os.path.basename(os.path.dirname(record['source']))}_{record['filename']}"
                    destination = os.path.join(image_dir, output_name)
                keep = augment is None or augment.keeps_original(split_name)
                pre_img, pre_tf = preprocess.original(record['source']) if preprocess else (None, None)
                src_w, src_h = image_width, image_height
                if pre_img is not None:
                    image_height, image_width = pre_img.shape[:2]
                if keep:
                    if pre_img is not None:
                        write_image(destination, pre_img)
                    else:
                        shutil.copy2(record['source'], destination)
            except (OSError, ValueError) as exc:
                print(f"[COCO] Skipping {record['source']}: {exc}")
                continue

            xml_path = os.path.join(xml_root, record['base'] + '.xml')
            annotations = read_coco_annotations(xml_path, src_w, src_h, categories)
            if keep:
                coco['images'].append({
                    'id': image_id,
                    'file_name': output_name,
                    'width': image_width,
                    'height': image_height
                })
                for annotation in annotations:
                    shown = annotation if pre_tf is None else coco_annotation(annotation, pre_tf)
                    shown['id'] = annotation_id
                    shown['image_id'] = image_id
                    coco['annotations'].append(shown)
                    annotation_id += 1
            else:
                image_id -= 1

            # Photometric copies keep the exact same boxes/polygons
            if augment is not None:
                stem, ext = os.path.splitext(output_name)
                for suffix, aug_img, tf in augment.copies(record['source'], split_name):
                    image_id += 1
                    aug_name = f"{stem}{suffix}{ext}"
                    try:
                        write_image(os.path.join(image_dir, aug_name), aug_img)
                    except (OSError, ValueError) as exc:
                        print(f"[COCO] Could not write {aug_name}: {exc}")
                        image_id -= 1
                        continue
                    coco['images'].append({
                        'id': image_id,
                        'file_name': aug_name,
                        'width': aug_img.shape[1],
                        'height': aug_img.shape[0]
                    })
                    for annotation in annotations:
                        clone = copy.deepcopy(annotation) if tf is None else coco_annotation(annotation, tf)
                        clone['id'] = annotation_id
                        clone['image_id'] = image_id
                        coco['annotations'].append(clone)
                        annotation_id += 1

            processed += 1
            if progress_cb:
                progress_cb(processed, total, f'Exporting COCO... {processed}/{total}')

        annotation_dir = os.path.join(target, 'annotations')
        os.makedirs(annotation_dir, exist_ok=True)
        json_path = os.path.join(annotation_dir, f'instances_{split_name}.json')
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(coco, json_file, indent=2)

    return target


def export_yolo(records, labels_dir, workspace, classes, target, train_pct, valid_pct, progress_cb=None, augment=None,
                preprocess=None, rng=random):
    """Export the workspace's YOLO labels as a train/val/test split, pulling
    source images straight from datasetsInput (never from a copied folder)."""
    images_by_base = {r['base']: r['source'] for r in records}

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] \
        if os.path.isdir(labels_dir) else []
    pairs = [(images_by_base[base], base + '.txt')
             for base in (os.path.splitext(f)[0] for f in label_files)
             if base in images_by_base]

    if not pairs:
        raise ValueError(f'No labeled images found for workspace {workspace}')

    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(round(n * (train_pct / 100.0)))
    n_valid = int(round(n * (valid_pct / 100.0)))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_valid],
        'test': pairs[n_train + n_valid:]
    }

    processed = 0
    for split_name, split_pairs in splits.items():
        out_images = os.path.join(target, split_name, 'images')
        out_labels = os.path.join(target, split_name, 'labels')
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)
        for img_src, lbl_name in split_pairs:
            keep = augment is None or augment.keeps_original(split_name)
            pre_img, pre_tf = preprocess.original(img_src) if preprocess else (None, None)
            if keep:
                try:
                    if pre_img is not None:
                        write_image(os.path.join(out_images, os.path.basename(img_src)), pre_img)
                    else:
                        shutil.copy2(img_src, os.path.join(out_images, os.path.basename(img_src)))
                except Exception:
                    pass
                try:
                    label_path = os.path.join(labels_dir, lbl_name)
                    if pre_tf is None:
                        shutil.copy2(label_path, os.path.join(out_labels, lbl_name))
                    else:
                        with open(label_path, encoding='utf-8') as f:
                            moved = yolo_label_text(f.read(), pre_tf)
                        with open(os.path.join(out_labels, lbl_name), 'w', encoding='utf-8') as f:
                            f.write(moved)
                except Exception:
                    pass

            # Augmented copies: photometric ones reuse the label as is, flips/rotations move it
            if augment is not None:
                stem, ext = os.path.splitext(os.path.basename(img_src))
                for suffix, aug_img, tf in augment.copies(img_src, split_name):
                    try:
                        write_image(os.path.join(out_images, f"{stem}{suffix}{ext}"), aug_img)
                        label_src = os.path.join(labels_dir, lbl_name)
                        label_dst = os.path.join(out_labels, f"{stem}{suffix}.txt")
                        if tf is None:
                            shutil.copy2(label_src, label_dst)
                        else:
                            with open(label_src, encoding='utf-8') as f:
                                moved = yolo_label_text(f.read(), tf)
                            with open(label_dst, 'w', encoding='utf-8') as f:
                                f.write(moved)
                    except Exception as exc:
                        print(f"[YOLO] Could not write augmented copy of {stem}: {exc}")

            processed += 1
            if progress_cb:
                progress_cb(processed, n, f'Exporting YOLO... {processed}/{n}')

    try:
        data_yml = os.path.join(target, 'data.yml')
        with open(data_yml, 'w') as fy:
            fy.write(f"train: {os.path.abspath(os.path.join(target, 'train', 'images'))}\n")
            fy.write(f"val:   {os.path.abspath(os.path.join(target, 'val', 'images'))}\n")
            fy.write(f"test:  {os.path.abspath(os.path.join(target, 'test', 'images'))}\n")
            fy.write(f"nc:    {len(classes)}\n")
            fy.write('names:\n')
            for i, name in enumerate(classes):
                fy.write(f"  {i}: {name}\n")
    except Exception:
        pass

    return target


def export_voc(records, voc_dir, workspace, target, train_pct, valid_pct, progress_cb=None, augment=None,
               preprocess=None, rng=random):
    """Export indexed workspace images and their VOC XML annotations."""
    images_found = [r['source'] for r in records]

    if not images_found:
        raise ValueError(f'No images found for workspace {workspace} in datasetsInput')

    rng.shuffle(images_found)
    n = len(images_found)
    n_train = int(round(n * (train_pct / 100.0)))
    n_valid = int(round(n * (valid_pct / 100.0)))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)

    train_imgs = images_found[:n_train]
    valid_imgs = images_found[n_train:n_train + n_valid]
    test_imgs = images_found[n_train + n_valid:]

    processed = 0
    for split_name, imgs_list in (('train', train_imgs), ('valid', valid_imgs), ('test', test_imgs)):
        out_dir = os.path.join(target, split_name)
        os.makedirs(out_dir, exist_ok=True)
        for img_path in imgs_list:
            keep = augment is None or augment.keeps_original(split_name)
            pre_img, pre_tf = preprocess.original(img_path) if preprocess else (None, None)
            base = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(voc_dir, base + '.xml')
            if keep:
                try:
                    if pre_img is not None:
                        write_image(os.path.join(out_dir, os.path.basename(img_path)), pre_img)
                    else:
                        shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))
                except Exception:
                    pass
                if os.path.exists(xml_path):
                    try:
                        if pre_tf is None:
                            shutil.copy2(xml_path, os.path.join(out_dir, base + '.xml'))
                        else:
                            voc_xml_write(xml_path, os.path.join(out_dir, base + '.xml'),
                                          os.path.basename(img_path), pre_tf)
                    except Exception:
                        pass

            # Augmented copies: same objects (moved with the image for flips/rotations)
            if augment is not None:
                img_stem, img_ext = os.path.splitext(os.path.basename(img_path))
                for suffix, aug_img, tf in augment.copies(img_path, split_name):
                    aug_name = f"{img_stem}{suffix}{img_ext}"
                    try:
                        write_image(os.path.join(out_dir, aug_name), aug_img)
                        if os.path.exists(xml_path):
                            voc_xml_write(xml_path, os.path.join(out_dir, f"{img_stem}{suffix}.xml"),
                                          aug_name, tf)
                    except Exception as exc:
                        print(f"[VOC] Could not write augmented copy of {img_stem}: {exc}")

            processed += 1
            if progress_cb:
                progress_cb(processed, n, f'Exporting Pascal VOC... {processed}/{n}')

    return target
