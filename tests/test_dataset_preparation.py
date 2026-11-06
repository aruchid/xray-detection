"""Regression coverage for stale split files and failed preparations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/train_alcast_yolo.py'
spec = importlib.util.spec_from_file_location('train_alcast_yolo', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'dataset'
        self.archive = self.root / 'input.zip'

    def write_archive(self, annotation='1 0.5 0.5 0.2 0.2'):
        with ZipFile(self.archive, 'w') as archive:
            archive.writestr('AlCastXray/Images/100.jpg', b'image fixture')
            archive.writestr('AlCastXray/Annotations/100.txt', annotation)

    def test_rebuild_removes_old_split_and_cache_files(self):
        self.write_archive()
        old = self.dataset / 'images/train/100.jpg'
        old.parent.mkdir(parents=True)
        old.write_bytes(b'stale')
        cache = self.dataset / 'labels/train.cache'
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b'stale cache')
        config = module.prepare(self.archive, self.dataset)
        self.assertFalse(old.exists())
        self.assertFalse(cache.exists())
        self.assertEqual(len(list(self.dataset.rglob('100.jpg'))), 1)
        self.assertEqual(json.loads(config.read_text())['path'], str(self.dataset.resolve()))
        label = next(self.dataset.rglob('100.txt'))
        self.assertEqual(label.read_text(), '0 0.5 0.5 0.2 0.2\n')

    def test_corrupt_archive_keeps_existing_dataset(self):
        self.dataset.mkdir()
        sentinel = self.dataset / 'alcast.yaml'
        sentinel.write_text('previous valid generation')
        self.archive.write_bytes(b'broken archive')
        with self.assertRaises(Exception):
            module.prepare(self.archive, self.dataset)
        self.assertEqual(sentinel.read_text(), 'previous valid generation')
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ['dataset', 'input.zip'])

    def test_concurrent_preparation_is_rejected(self):
        self.write_archive()
        lock = self.dataset.with_name('dataset.lock')
        lock.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'already locked'):
            module.prepare(self.archive, self.dataset)
        self.assertTrue(lock.exists())


if __name__ == '__main__':
    unittest.main()
