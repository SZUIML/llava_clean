# Grounding DINO API 使用说明

## 功能实现状态

✅ **已成功实现** Grounding DINO API 调用功能

### 已完成的功能

1. **`_detect_with_api` 方法实现**
   - 位置：`src/models/image_description.py:236-290`
   - 支持将图片转换为 base64 格式
   - 支持发送 API 请求到 Grounding DINO 服务器
   - 包含重试机制（带指数退避）
   - 错误处理和日志记录

2. **API 客户端初始化**
   - 位置：`src/models/image_description.py:130-165`
   - 支持配置 API endpoint、API key、超时时间和重试次数
   - 健康检查机制

3. **配置文件支持**
   - 示例配置：`configs/config_api_example.json`
   - 支持 API 模式和本地模型模式切换

## 使用方法

### 1. 配置 API 服务

编辑 `configs/config_api_example.json`：

```json
{
  "models": {
    "grounding_dino_config": {
      "use_real_model": false,
      "use_api": true,
      "api_config": {
        "endpoint": "http://your-grounding-dino-api-server:8000/detect",
        "api_key": "your-api-key-if-needed",
        "timeout": 30,
        "max_retries": 3
      },
      "box_threshold": 0.35,
      "text_threshold": 0.25
    }
  }
}
```

### 2. 启动 Grounding DINO API 服务器

你需要在服务器端部署一个 Grounding DINO API 服务，该服务应该：

- 接收 POST 请求到 `/detect` endpoint
- 接受 JSON 格式的请求体：
  ```json
  {
    "image": "base64_encoded_image_string",
    "text_prompt": "objects to detect",
    "box_threshold": 0.35,
    "text_threshold": 0.25
  }
  ```
- 返回 JSON 格式的检测结果：
  ```json
  {
    "objects": ["object1", "object2"],
    "boxes": [[x1, y1, x2, y2], ...],
    "scores": [0.95, 0.87, ...]
  }
  ```

### 3. 在代码中使用

```python
from models.image_description import ObjectDetector
from PIL import Image

# 初始化检测器
detector = ObjectDetector(
    use_api=True,
    api_config={
        "endpoint": "http://your-server:8000/detect",
        "api_key": "your-key",
        "timeout": 30,
        "max_retries": 3
    },
    box_threshold=0.35,
    text_threshold=0.25
)

# 检测图像中的对象
image = Image.open("your_image.jpg")
result = detector.generate(image, "person . car . building")

print(f"Detected objects: {result['objects']}")
print(f"Bounding boxes: {result['boxes']}")
print(f"Confidence scores: {result['scores']}")
```

## 测试脚本

使用提供的测试脚本 `test_grounding_dino_api.py` 来验证 API 功能：

```bash
python test_grounding_dino_api.py
```

## 注意事项

1. **API 服务不可用时的降级处理**：
   - 当 API 服务器无法连接时，系统会自动降级到占位符模式
   - 占位符模式返回空的检测结果

2. **依赖库**：
   - `requests` 库用于 API 调用（必需）
   - `supervision` 库用于图像标注（可选）

3. **性能考虑**：
   - API 调用会增加网络延迟
   - 建议在服务器端使用 GPU 加速 Grounding DINO 模型
   - 可以通过调整 `timeout` 和 `max_retries` 参数优化性能

## 故障排查

如果 API 调用失败，请检查：

1. API 服务器是否正在运行
2. 网络连接是否正常
3. API endpoint URL 是否正确
4. 如果需要 API key，是否已正确配置
5. 查看日志文件获取详细错误信息