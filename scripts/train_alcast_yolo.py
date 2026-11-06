"""Train YOLO on the fixed Al-Cast split and evaluate the best checkpoint."""
import argparse
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(archive_path, dataset, final_path):
    split_file = ROOT / 'artifacts/alcast-yolo/split.json'
    groups = json.loads(split_file.read_text())['split_id_blocks']
    split_by_block = {block: split for split, blocks in groups.items() for block in blocks}
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            path = PurePosixPath(name)
            if path.parent.name != 'Images' or path.suffix.lower() != '.jpg':
                continue
            image_id = int(path.stem)
            split = split_by_block[image_id // 50]
            image = dataset / 'images' / split / f'{image_id}.jpg'
            label = dataset / 'labels' / split / f'{image_id}.txt'
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(archive.read(name))
            annotation = str(path.parent.parent / 'Annotations' / f'{image_id}.txt')
            rows = archive.read(annotation).decode().splitlines() if annotation in names else []
            # Localize any defect as one class while retaining each original box.
            label.write_text(''.join('0 ' + ' '.join(row.split()[1:]) + '\n'
                                     for row in rows if row.strip()))
    config = dataset / 'alcast.yaml'
    config.write_text(json.dumps({'path': str(final_path), 'train': 'images/train',
                                  'val': 'images/validation', 'test': 'images/test',
                                  'names': ['defect']}, indent=2))
    return config


@contextmanager
def preparation_lock(dataset):
    lock = dataset.with_name(dataset.name + '.lock')
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f'Dataset preparation already locked: {lock}') from error
    try:
        yield
    finally:
        lock.rmdir()


def prepare(archive_path, dataset):
    """Publish a complete replacement, preserving the prior dataset on failure."""
    dataset = dataset.resolve()
    dataset.parent.mkdir(parents=True, exist_ok=True)
    with preparation_lock(dataset):
        staging = Path(tempfile.mkdtemp(prefix=dataset.name + '-staging-', dir=dataset.parent))
        backup = staging / 'previous'
        try:
            built = staging / 'dataset'
            _write_dataset(archive_path, built, dataset)
            if dataset.exists():
                dataset.rename(backup)
            try:
                built.rename(dataset)
            except BaseException:
                if backup.exists():
                    backup.rename(dataset)
                raise
        finally:
            shutil.rmtree(staging)
    return dataset / 'alcast.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    workspace = ROOT / 'data/yolo'
    config = prepare(args.archive, workspace / 'dataset_dedup_v2')
    model = YOLO(str(workspace / 'yolo11n.pt'))
    model.train(data=str(config), device=args.device, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, workers=0, seed=42,
                amp=False, project=str(workspace), name='train_dedup_20e_640', close_mosaic=5)
    best = Path(model.trainer.save_dir) / 'weights/best.pt'
    metrics = YOLO(str(best)).val(data=str(config), split='test', device=args.device,
                                  imgsz=args.imgsz, batch=args.batch, workers=0,
                                  project=str(workspace), name='test_dedup_20e_640', plots=True)
    result = {key: float(value) for key, value in metrics.results_dict.items()}
    result.update(epochs=args.epochs, imgsz=args.imgsz, device=args.device,
                  checkpoint=str(best.relative_to(ROOT)))
    (workspace / 'test-metrics-20e-640.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
