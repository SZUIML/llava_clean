# LLaVA-CoT Dataset Cleaning Pipeline

基于 R1-Onevision 论文方法的 LLaVA-CoT 数据集清洗项目。该项目通过生成形式化图像描述、重构思维链（CoT）和质量控制来提升数据集质量。支持从大规模数据集中智能采样高质量子集。

## 主要特性

- 🔍 **形式化图像描述生成**：结合VLM、OCR和目标检测
- 🤔 **CoT重构**：基于形式化描述重写思维链
- ✅ **质量控制**：多维度质量评估和过滤
- 📊 **智能采样**：从100k数据中选择50k高质量样本
- 🔄 **批处理**：支持大规模数据处理
- 💾 **断点续传**：中间结果保存，支持失败恢复

## 项目结构

```
llava_clean/
├── src/
│   ├── models/
│   │   └── image_description.py      # 图像形式化描述生成
│   ├── processors/
│   │   ├── cot_restructure.py        # CoT重构和答案提取
│   │   ├── quality_check.py          # 质量检查和过滤
│   │   └── data_sampler.py           # 智能数据采样
│   └── utils/
│       └── data_loader.py            # 数据加载和处理
├── configs/
│   └── config_example.json           # 配置文件示例
├── data/
│   ├── input/                        # 输入数据目录
│   └── output/                       # 输出数据目录
├── clean_llava_cot.py                # 主处理脚本
├── requirements.txt                   # 依赖包
└── README.md                          # 本文档
```

## 安装

1. 克隆项目到本地或服务器：
```bash
git clone <your-repo-url>
cd llava_clean
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 如果使用 OCR 功能，可能需要额外安装：
```bash
# EasyOCR 可能需要额外的系统依赖
pip install easyocr
```

## 配置

1. 复制配置文件模板：
```bash
cp configs/config_example.json configs/config.json
```

2. 编辑 `configs/config.json`，修改以下关键配置：

### 必须修改的配置：
- `data.input_path`: LLaVA-CoT 数据集路径（.json 或 .jsonl）
- `data.image_dir`: 图像文件目录路径
- `api_keys.openai`: 你的 OpenAI API Key
- `output.output_path`: 清洗后数据集的输出路径

### 可选配置：
- `data.max_samples`: 限制处理样本数（用于测试）
- `processing.batch_size`: 批处理大小
- `processing.api_delay`: API 调用间隔（避免限流）
- `quality.min_score`: 质量分数阈值（0-1）
- `quality.keep_failed_samples`: 是否保留未通过质量检查的样本

## 使用方法

### 基本运行：
```bash
python clean_llava_cot.py --config configs/config.json
```

### 带日志文件运行：
```bash
python clean_llava_cot.py --config configs/config.json --log-file logs/cleaning.log
```

### 调试模式：
```bash
python clean_llava_cot.py --config configs/config.json --log-level DEBUG
```

### 数据量控制选项：

如果你想从 LLaVA-CoT-100k 中只处理前 50k 数据（不经过复杂的采样算法），可以在配置文件中设置：

```json
{
  "data": {
    "limit_to_first_half": true,
    "max_samples": null
  },
  "sampling": {
    "enabled": false
  }
}
```

或者直接指定具体数量：
```json
{
  "data": {
    "limit_to_first_half": false,
    "max_samples": 50000
  },
  "sampling": {
    "enabled": false
  }
}
```

**选项说明**：
- `limit_to_first_half: true` - 只处理数据集的前一半
- `max_samples: 50000` - 只处理前 50000 个样本
- `sampling.enabled: false` - 禁用智能采样功能

## 数据格式

### 输入格式（LLaVA-CoT）
支持多种格式，包括：
```json
{
  "id": "sample_001",
  "image": "image_001.jpg",
  "question": "What is shown in the image?",
  "rationale": "The image shows...",
  "answer": "A cat"
}
```

或对话格式：
```json
{
  "id": "sample_001",
  "image": "image_001.jpg",
  "conversations": [
    {"from": "human", "value": "What is shown?"},
    {"from": "gpt", "value": "<think>...</think>Answer"}
  ]
}
```

### 输出格式
```json
{
  "id": "sample_001",
  "image_path": "/path/to/image_001.jpg",
  "question": "What is shown in the image?",
  "image_formal_description": "The image shows a pulley system...",
  "cot_thinking": "<think>According to the description...</think>",
  "final_answer": "<answer>60N</answer>",
  "quality_metrics": {
    "overall_score": 0.85,
    "issues": []
  },
  "passed_quality_check": true
}
```

## 处理流程

1. **数据加载**: 读取 LLaVA-CoT 数据集和对应图像
2. **图像分析**: 
   - 识别图像类型（自然场景、图表、数学等）
   - 生成密集描述（Dense Caption）
   - 提取对象和文本（可选）
3. **形式化描述生成**: 综合所有信息生成结构化描述
4. **CoT 重构**: 基于形式化描述重写思维链
5. **答案提取**: 从 CoT 中提取并格式化最终答案
6. **质量检查**: 评估描述、CoT 和答案的质量
7. **数据过滤**: 根据质量分数决定保留或丢弃

## 部署到服务器

1. 上传代码到服务器：
```bash
scp -r llava_clean/ user@server:/path/to/destination/
```

2. 在服务器上配置环境：
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

3. 修改配置文件中的路径：
```bash
vim configs/config.json
# 修改所有路径为服务器上的实际路径
```

4. 使用 screen 或 tmux 后台运行：
```bash
screen -S llava_cleaning
python clean_llava_cot.py --config configs/config.json
# Ctrl+A+D 分离会话
```

## 性能优化

1. **批处理**: 调整 `batch_size` 以优化内存使用
2. **API 限流**: 设置适当的 `api_delay` 避免触发限流
3. **并行处理**: 可以将数据集分割后并行处理
4. **中间保存**: 启用 `save_intermediate` 以便从失败中恢复

## 故障排除

### 常见问题

1. **API Key 错误**: 
   - 确保在配置文件中正确设置了 OpenAI API Key
   - 检查 API Key 是否有效且有足够配额

2. **EasyOCR 网络连接问题**:
   ```
   ERROR - Failed to initialize OCR: <urlopen error [Errno 104] Connection reset by peer>
   ```
   
   **解决方案**：
   - **临时禁用OCR**：在配置文件中设置 `"use_ocr": false`
   - **使用OCR容错模式**：设置 `"ocr_fallback_on_error": true`（默认已启用）
   - **手动下载模型**：
     ```bash
     python -c "import easyocr; easyocr.Reader(['en'])"
     ```
   - **使用代理或VPN**：如果网络受限
   - **离线环境**：可以先在有网络的环境下初始化EasyOCR，然后复制模型文件

3. **图像加载失败**:
   - 检查 `image_dir` 路径是否正确
   - 确保图像文件存在且格式支持

4. **内存不足**:
   - 减小 `batch_size`
   - 使用 `max_samples` 限制处理数量
   - 禁用不必要的功能（如 `use_object_detection: false`）

5. **处理速度慢**:
   - 考虑使用更快的模型（如 gpt-3.5-turbo）
   - 增加 `batch_size`（如果内存允许）
   - 禁用OCR和目标检测以减少处理时间

### OCR相关配置建议

如果遇到OCR问题，推荐的配置：

```json
{
  "models": {
    "use_object_detection": false,
    "use_ocr": false,
    "ocr_fallback_on_error": true
  }
}
```

这样仍能获得高质量的形式化描述，只是缺少OCR文本提取功能。

## 输出统计

处理完成后会生成 `processing_stats.json`，包含：
- 总样本数
- 处理成功/失败数
- 质量检查通过率
- 常见失败原因
- 处理时间

## 扩展功能

项目设计为模块化架构，易于扩展：

1. **添加新的图像分析模型**: 修改 `src/models/image_description.py`
2. **自定义质量检查规则**: 修改 `src/processors/quality_check.py`
3. **支持新的数据格式**: 修改 `src/utils/data_loader.py`

## 注意事项

- API 调用会产生费用，建议先用小数据集测试
- 确保有足够的磁盘空间存储处理后的数据
- 对于大规模数据集，建议分批处理并定期保存
- 敏感数据处理时注意隐私保护

## License

MIT

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。