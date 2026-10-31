# 验证记录

`python -m unittest discover -s tests -v`

```text
test_invalid_data (test_prototype.PrototypeTests.test_invalid_data) ... ok
test_reference_recovery (test_prototype.PrototypeTests.test_reference_recovery) ... ok
test_reject_invalid_process_input (test_prototype.PrototypeTests.test_reject_invalid_process_input) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.022s

OK
```

`python src/prototype.py --output artifacts` 生成实际结果图及 metrics.json。指标来自固定种子合成样例，不代表真实场景精度。
