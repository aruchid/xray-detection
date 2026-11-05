"""Export a fixed first-three-test-image comparison using the trained model."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    args = parser.parse_args()
    images = sorted((ROOT / 'data/yolo/dataset_dedup_v2/images/test').glob('*.jpg'),
                    key=lambda path: int(path.stem))[:3]
    model = YOLO(str(args.weights))
    fig, axes = plt.subplots(3, 2, figsize=(10, 13), layout='constrained')
    for path, (left, right) in zip(images, axes):
        image = Image.open(path).convert('RGB')
        width, height = image.size
        for ax in (left, right):
            ax.imshow(image)
            ax.axis('off')
        label = ROOT / 'data/yolo/dataset_dedup_v2/labels/test' / (path.stem + '.txt')
        for line in label.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            left.add_patch(Rectangle(((cx-bw/2)*width, (cy-bh/2)*height), bw*width,
                                     bh*height, fill=False, edgecolor='lime', linewidth=1.5))
        result = model.predict(str(path), imgsz=640, conf=.25, iou=.7,
                               device='mps', verbose=False)[0]
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            right.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                      edgecolor='red', linewidth=1.5))
            right.text(x1, y1, f'{conf:.2f}', color='red', fontsize=8)
        left.set_title(f'{path.name} · reference boxes')
        right.set_title('YOLO11n predictions · confidence ≥ 0.25')
    fig.suptitle('Al-Cast real X-ray images · first 3 test IDs · 20 epochs')
    output = ROOT / 'artifacts/alcast-yolo/predictions.png'
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
