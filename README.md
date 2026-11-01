# X 光异常区域基线

## 当前实现

基于残差 MAD 稳健尺度估计输出异常掩膜与分数。

这是可运行的数值基线原型，不等同于网站介绍中的完整系统。

## 输入契约

已配准灰度图像与同形状正常参考图。核心函数位于 `src/prototype.py` 的 `process`，可直接导入调用。默认命令运行固定种子合成样例，并在代码中断言预期效果；真实数据需由调用者按照函数形状约束传入。

## 数值有效性与阈值

`process(image, reference, threshold=5.0)` 要求二维、非空、同形状且有限的输入。阈值必须为有限的正标量；NaN、无穷、零、负值或向量均抛出 `ValueError`。即使输入有限，极端数值也可能使残差或评分溢出，此时明确拒绝计算，避免无效分数被当成“没有异常”。需在调用前对图像和参考使用相同尺度进行合理缩放。

评分为 `abs(residual - median(residual)) / max(1.4826 * MAD, 1e-8)`；仅 `score > threshold` 判为异常，等于阈值不标记。MAD 为零时使用固定尺度下限，结果依赖输入灰度尺度。整体恒定偏移会被残差中位数消除，因此本方法不能检测全图统一亮度偏移。

## 快速运行

Python 3.11+。

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/prototype.py --output artifacts
python -m unittest discover -s tests -v
```

## 输出与验证

- `artifacts/result.png`：当前代码生成的结果可视化。
- `artifacts/metrics.json`：样例指标。
- `artifacts/results.npz`：结果数组，按图例顺序编号。

![实际运行结果（合成样例）](artifacts/result.png)

```json
{
  "synthetic_recall": 1.0,
  "false_positive_pixels": 0
}
```

固定种子回归检查与异常数据检查已运行。图中数据均为合成，不是硬件实测，也不是应用界面截图。

## 需求、架构与边界

[PRD](docs/PRD.md) · [当前原型范围](docs/IMPLEMENTATION.md) · [验证记录](docs/VALIDATION.md)

仅工业图像算法演示，不适用于医学诊断；需要正常参考和配准，未训练目标检测网络。

## 后续里程碑

1. 使用真实输入建立基准数据及明确误差界限。
2. 补充数据导入和交互界面，按需求接入 SDK、设备或模型。
3. 记录独立测试集、资源消耗和失败案例，再评估部署条件。
