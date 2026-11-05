"""Evaluate an annotation-supervised localization baseline on the Al-Cast ZIP."""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageFilter

SIZE = 384
GRID = np.arange(1, SIZE, 3)
Y, X = np.meshgrid((GRID + 0.5) / SIZE, (GRID + 0.5) / SIZE, indexing='ij')


def load_sample(archive, image_path, boxes):
    with Image.open(io.BytesIO(archive.read(image_path))) as image:
        image = image.convert('L').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    gray = np.asarray(image, dtype=np.float32) / 255
    features = [gray]
    for radius in (2, 5, 12):
        smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255
        residual = gray - smooth
        dy, dx = np.gradient(smooth)
        features.extend([residual, np.abs(residual), np.hypot(dx, dy)])
    features = np.stack(features, axis=-1)[GRID[:, None], GRID].reshape(X.size, -1)
    inside = np.zeros(X.shape, dtype=bool)
    for cx, cy, width, height in boxes:
        inside |= (np.abs(X - cx) <= width / 2) & (np.abs(Y - cy) <= height / 2)
    return gray, features, inside.ravel()


def evaluate(archive, samples, model):
    mean, scale, weights = model
    results = []
    for image_path, boxes in samples:
        _, features, inside = load_sample(archive, image_path, boxes)
        scores = ((features - mean) / scale) @ weights
        peak = int(scores.argmax())
        results.append({'image': image_path, 'defective': bool(boxes),
                        'score': float(scores[peak]), 'peak_in_box': bool(inside[peak]),
                        'box_grid_fraction': float(inside.mean())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/alcast-run'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    with ZipFile(args.archive) as archive:
        annotations = {PurePosixPath(name).stem: name for name in archive.namelist()
                       if '/Annotations/' in name and name.endswith('.txt')}
        samples = []
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} or path.parent.name != 'Images':
                continue
            boxes = []
            label = annotations.get(path.stem)
            for line in archive.read(label).decode().splitlines() if label else []:
                if line.strip():
                    _, cx, cy, width, height = map(float, line.split())
                    boxes.append((cx, cy, width, height))
            samples.append((name, boxes))
        image_ids = {PurePosixPath(name).stem for name, _ in samples}
        if annotations.keys() - image_ids:
            raise ValueError('Annotations without a matching image')
        samples.sort(key=lambda item: int(PurePosixPath(item[0]).stem))
        if len({PurePosixPath(name).stem for name, _ in samples}) != len(samples):
            raise ValueError('Duplicate image IDs')
        groups = sorted({int(PurePosixPath(name).stem) // 50 for name, _ in samples})
        rng.shuffle(groups)
        partitions = [set(part.tolist()) for part in np.split(np.asarray(groups), [int(.6 * len(groups)), int(.8 * len(groups))])]
        train, validation, test = [[item for item in samples if int(PurePosixPath(item[0]).stem) // 50 in group]
                                   for group in partitions]
        for split in (train, validation, test):
            if not any(boxes for _, boxes in split) or not any(not boxes for _, boxes in split):
                raise ValueError('Each split requires images with and without annotated boxes')
        positive, negative = [], []
        for name, boxes in train:
            _, features, inside = load_sample(archive, name, boxes)
            for indices, destination, limit in [(np.flatnonzero(inside), positive, 32),
                                                (np.flatnonzero(~inside), negative, 64)]:
                if indices.size:
                    destination.append(features[rng.choice(indices, min(limit, indices.size), replace=False)])
        pos, neg = np.concatenate(positive), np.concatenate(negative)
        features = np.concatenate([pos, neg]).astype(float)
        mean = features.mean(axis=0)
        scale = np.maximum(features.std(axis=0), 1e-6)
        pos, neg = (pos - mean) / scale, (neg - mean) / scale
        covariance = (pos.T @ pos / len(pos) + neg.T @ neg / len(neg)) / 2
        weights = np.linalg.solve(covariance + 0.1 * np.eye(len(mean)), (pos.mean(axis=0) - neg.mean(axis=0)) / 2)
        model = mean, scale, weights
        val_results = evaluate(archive, validation, model)
        threshold = float(np.quantile([row['score'] for row in val_results if not row['defective']], .95, method='higher'))
        results = evaluate(archive, test, model)
        positives = [row for row in results if row['defective']]
        negatives = [row for row in results if not row['defective']]
        comparisons = np.array([r['score'] for r in positives])[:, None] - np.array([r['score'] for r in negatives])
        metrics = {'splits': {name: {'images': len(split), 'defective': sum(bool(boxes) for _, boxes in split)}
                              for name, split in zip(('train', 'validation', 'test'), (train, validation, test))},
                   'threshold': threshold, 'image_auroc': float(np.mean((comparisons > 0) + .5 * (comparisons == 0))),
                   'test_recall': sum(r['score'] > threshold for r in positives) / len(positives),
                   'test_false_positive_rate': sum(r['score'] > threshold for r in negatives) / len(negatives),
                   'peak_in_box_rate': sum(r['peak_in_box'] for r in positives) / len(positives),
                   'mean_box_grid_fraction': float(np.mean([r['box_grid_fraction'] for r in positives]))}
        (args.output / 'report.json').write_text(json.dumps({'metrics': metrics, 'split_id_blocks': {
            name: sorted(group) for name, group in zip(('train', 'validation', 'test'), partitions)},
            'negative_definition': 'No defect box in the released annotations',
            'test': results}, indent=2) + '\n')
        np.savez(args.output / 'model.npz', mean=mean, scale=scale, weights=weights, threshold=threshold)
        lookup = dict(test)
        examples = [next((r for r in positives if r['peak_in_box']), positives[0]),
                    next((r for r in positives if not r['peak_in_box']), positives[-1]), negatives[0]]
        fig, axes = plt.subplots(3, 2, figsize=(8, 10), layout='constrained')
        for row, (left, right) in zip(examples, axes):
            gray, features, _ = load_sample(archive, row['image'], lookup[row['image']])
            scores = (((features - mean) / scale) @ weights).reshape(X.shape)
            left.imshow(gray, cmap='gray', extent=(0, 1, 1, 0))
            right.imshow(scores, cmap='magma', extent=(0, 1, 1, 0))
            for cx, cy, width, height in lookup[row['image']]:
                left.add_patch(Rectangle((cx-width/2, cy-height/2), width, height, fill=False, edgecolor='lime'))
            peak = np.unravel_index(scores.argmax(), scores.shape)
            right.plot(X[peak], Y[peak], 'c+')
            left.set_title(PurePosixPath(row['image']).name + ' · reference boxes')
            right.set_title('Anomaly score · peak')
            left.axis('off')
            right.axis('off')
        fig.savefig(args.output / 'preview.png', dpi=150)
        plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
