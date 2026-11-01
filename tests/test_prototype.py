import sys,unittest
import warnings
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import prototype as p
class PrototypeTests(unittest.TestCase):
 def test_reference_recovery(self):
  _,_,metrics=p.demo()
  self.assertTrue(metrics)
 def test_invalid_data(self):
  with self.assertRaises(ValueError):p.finite([float('nan')])
  with self.assertRaises(ValueError):p.finite([])
 def test_reject_invalid_process_input(self):
  with self.assertRaises(ValueError):p.process([[1]],[[1,2]])

 def test_nonfinite_and_nonscalar_thresholds_are_rejected(self):
  image = np.array([[0., 0.], [0., 10.]])
  for threshold in [float('nan'), float('inf'), -float('inf'), 0, -1,
                    [5], np.array([5, 6]), None, complex(1, 2)]:
   with self.subTest(threshold=repr(threshold)):
    with self.assertRaisesRegex(ValueError, 'Threshold'):
     p.process(image, np.zeros_like(image), threshold)

 def test_finite_inputs_with_overflow_fail_explicitly(self):
  limit = np.finfo(float).max
  cases = [
   ([[limit, 0]], [[-limit, 0]]),  # subtraction overflow
   ([[0., 0., 0., limit]], [[0., 0., 0., 0.]]),  # score overflow
  ]
  with warnings.catch_warnings():
   warnings.simplefilter('error', RuntimeWarning)
   for image, reference in cases:
    with self.subTest(image=image):
     with self.assertRaisesRegex(ValueError, 'floating-point range'):
      p.process(image, reference)

 def test_signed_outliers_and_strict_threshold(self):
  image = np.array([[0., 0., 0.], [0., 2., -2.]])
  mask, score = p.process(image, np.zeros_like(image))
  np.testing.assert_array_equal(mask, [[False, False, False], [False, True, True]])
  self.assertTrue(np.isfinite(score).all())
  self.assertEqual(score[1, 1], score[1, 2])
  at_boundary, _ = p.process(image, np.zeros_like(image), score[1, 1])
  self.assertFalse(at_boundary.any())

 def test_constant_residual_remains_finite(self):
  mask, score = p.process(np.full((3, 3), 7.), np.full((3, 3), 2.))
  self.assertFalse(mask.any())
  np.testing.assert_array_equal(score, np.zeros((3, 3)))
