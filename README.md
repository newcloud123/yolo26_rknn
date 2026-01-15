# YOLO26 RKNN 模型转换与推理

## 项目简介
本项目提供了 YOLO26 模型从训练到 RKNN 格式转换及推理的完整流程，支持 Python 端推理和 C++ 端推理，适用于 Rockchip 系列芯片（如 RK3588）。

## 目录结构
```
yolo26_rknn/
├── yolo26/              # YOLO26 模型源码目录
├── onnx2rknn.py         # ONNX 转 RKNN 及推理脚本
├── requirements.txt     # 依赖包列表
└── README.md            # 项目说明文档
```

## 1. 环境准备
安装所需依赖包：
```bash
pip install -r requirements.txt
```

## 2. 训练 YOLO26 模型
通过 Ultralytics 官方仓库训练 YOLO26 模型：

1. 克隆 Ultralytics 仓库：
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   ```

2. 按照 Ultralytics 官方文档训练 YOLO26 模型

3. 将训练好的模型权重文件（如 `best.pt`）复制到 `yolo26/` 目录下

## 3. 转换模型为 ONNX 格式
将训练好的 YOLO26 模型转换为 ONNX 格式：

```bash
cd yolo26_rknn/yolo26
python export_onnx.py
```

**注意事项：**
- 确保 `export.py` 脚本中的模型路径和参数设置正确
- 默认输出 ONNX 文件名为 `yolo26.onnx`

## 4. 转换模型为 RKNN 格式并推理
将 ONNX 模型转换为 RKNN 格式并进行 Python 端推理：

```bash
cd yolo26_rknn
python onnx2rknn.py
```

**脚本功能说明：**
- 加载 ONNX 模型
- 配置 RKNN 模型参数（如均值、标准差、量化方法等）
- 构建并导出 RKNN 模型
- 初始化 RKNN 运行时环境
- 执行推理并可视化结果

**关键参数配置：**
- `ONNX_MODEL`：ONNX 模型路径（默认：`./lsy.onnx`）
- `RKNN_MODEL`：RKNN 模型输出路径（默认：`./yolo26s.float.rknn`）
- `QUANTIZE_ON`：是否进行量化（默认：`False`）
- `target_platform`：目标平台（默认：`rk3588`）

## 5. C++ 端推理
对于需要在 C++ 环境下进行推理的用户，可以参考以下仓库：

```
https://github.com/newcloud123/yolo26-rk3588-cpp.git
```

该仓库提供了基于 RKNN API 的 C++ 推理实现，适用于 RK3588 等 Rockchip 平台。

## 注意事项

1. **模型路径设置**：
   - 确保 `onnx2rknn.py` 脚本中的模型路径与实际文件路径一致
   - 默认使用 `lsy.onnx` 作为输入模型，可根据实际情况修改

2. **推理图像**：
   - 默认使用 `lsy151.jpg` 作为推理图像
   - 可替换为自己的测试图像，确保图像路径正确

3. **目标平台**：
   - 默认目标平台为 `rk3588`
   - 如需在其他 Rockchip 平台使用，可修改 `target_platform` 参数

4. **量化设置**：
   - 默认关闭量化（`QUANTIZE_ON=False`）
   - 如需开启量化，需准备量化数据集（`DATASET` 参数）

5. **结果可视化**：
   - 推理结果会保存为 `test_rknn_result.jpg`
   - 包含检测框和类别置信度

## 常见问题

1. **转换失败**：
   - 检查 ONNX 模型是否正确生成
   - 确保 RKNN Toolkit2 版本与目标平台兼容

2. **推理结果不准确**：
   - 检查图像预处理是否正确
   - 调整检测阈值（`objectThresh` 参数）
   - 考虑使用量化模型提高性能

3. **运行时错误**：
   - 确保 RKNN 运行时环境正确安装
   - 检查设备是否支持目标平台

## 联系方式
如有问题或建议，欢迎提交 Issue 或 Pull Request。

## 许可证
本项目基于 Ultralytics YOLO26 模型，遵循相关开源许可证。
