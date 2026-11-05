"""Reproducible real casting-patch classification diagnostic; no synthetic defects."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prototype import process
ROOT = Path(__file__).resolve().parents[1]
COMMIT = '30de3af0edee237ac3fb7838a14d9f156a7e2fa5'
BASE = f'https://raw.githubusercontent.com/computervision-xray-testing/pyxvis/{COMMIT}/'

def fetch(path):
    target = ROOT / 'data/pyxvis' / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        content = subprocess.check_output(['curl', '-fsSL', '--retry', '3', '--max-time', '45', BASE + path])
        Image.open(io.BytesIO(content)).verify()
        target.write_bytes(content)
    content = target.read_bytes()
    with Image.open(target) as im:
        array = np.asarray(im.convert('L'), dtype=float) / 255
    return array, {'path': path, 'url': BASE + path, 'sha256': hashlib.sha256(content).hexdigest(), 'shape': list(array.shape)}

def gradient_features(image):
    """Pool unsigned edge strength into a 4×4 grid."""
    dy, dx = np.gradient(image)
    magnitude = np.hypot(dx, dy)
    pooled = magnitude.reshape(4, 8, 4, 8).mean(axis=(1, 3)).ravel()
    return pooled / max(np.linalg.norm(pooled), 1e-12)


def main():
    paths = json.loads((ROOT / 'scripts/real_data_paths.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        downloaded = list(pool.map(fetch, paths))
    images = [v[0] for v in downloaded]
    if any(im.shape != (32, 32) for im in images):
        raise ValueError('Expected 32×32 casting patches')
    templates = [np.median(images[20*c:20*(c+1)], axis=0) for c in range(2)]
    # Class-labelled training templates: experimental classifier, not a paired normal reference.
    training_features = np.stack([gradient_features(image) for image in images[:40]])
    training_labels = np.repeat([0, 1], 20)
    gradient_confusion = np.zeros((2, 2), dtype=int)
    rows = []
    confusion = np.zeros((2, 2), dtype=int)
    for i, image in enumerate(images[40:]):
        label = i // 20
        outputs = [process(image, ref, threshold=5.0) for ref in templates]
        distances = [float(np.mean(np.minimum(score, 20))) for _, score in outputs]
        predicted = int(np.argmin(distances))
        confusion[label, predicted] += 1
        gradient_distances = np.linalg.norm(training_features - gradient_features(image), axis=1)
        neighbors = np.argsort(gradient_distances, kind='stable')[:3]
        gradient_prediction = int(training_labels[neighbors].sum() >= 2)
        gradient_confusion[label, gradient_prediction] += 1
        rows.append({'path': paths[40+i], 'class_id': label, 'predicted_class_id': predicted,
                     'template_distances': distances, 'gradient_knn_class_id': gradient_prediction,
                     'flagged_fraction_against_class0': float(outputs[0][0].mean())})
    out = ROOT / 'artifacts/real-data'
    out.mkdir(parents=True, exist_ok=True)
    report = {'dataset': 'Author PyXVis casting JPEG patches', 'source_commit': COMMIT,
              'train_count': 40, 'test_count': 40, 'shape': list(images[0].shape),
              'selection': 'lexicographic first 20 per source split and class; exploratory subset',
              'threshold': 5.0, 'score_clip': 20, 'accuracy': float(np.trace(confusion)/40),
              'confusion_matrix_true_rows_pred_columns': confusion.tolist(),
              'gradient_knn': {'neighbors': 3, 'accuracy': float(np.trace(gradient_confusion) / 40),
                               'confusion_matrix_true_rows_pred_columns': gradient_confusion.tolist()},
              'class_semantics': 'source class_0/class_1; normal/defect mapping not independently verified',
              'limitation': 'Unregistered patches, no paired normal reference or pixel truth; exploratory class-ID classification only. Not defect-localization or field acceptance.',
              'samples': [v[1] for v in downloaded], 'results': rows}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, matrix, name in zip(axes, [confusion, gradient_confusion], ['Residual templates', 'Gradient 3-NN']):
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=20)
        for (row, col), count in np.ndenumerate(matrix):
            ax.text(col, row, str(count), ha='center', va='center',
                    color='white' if count > 12 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['class_0', 'class_1'],
               yticklabels=['class_0', 'class_1'], xlabel='Predicted class',
               ylabel='Source class', title=f'{name}: {np.trace(matrix)}/40 correct')
    fig.savefig(out/'summary.png', dpi=150)
    plt.close(fig)
    # Restricted source images and derived image gallery stay in ignored local data directory.
    fig, axes = plt.subplots(4,3, figsize=(10,10), layout='constrained')
    for r, idx in enumerate([40,41,60,61]):
        mask,score=process(images[idx], templates[0])
        for ax, arr, title in zip(axes[r], [images[idx],mask,score], ['Real input','Template residual mask','Residual score']):
            ax.imshow(arr,cmap='gray' if title=='Real input' else 'viridis');ax.set_title(f'{title} / class_{(idx-40)//20}');ax.axis('off')
    fig.savefig(ROOT/'data/real-data-preview.png', dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['train_count','test_count','shape','accuracy','confusion_matrix_true_rows_pred_columns','gradient_knn']},indent=2))

if __name__=='__main__': main()
