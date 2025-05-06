# Triton-Inference-Server

## 🧠 Triton Model Management

To manually load or unload models during runtime, make sure the Triton Inference Server is launched with `--model-control-mode=explicit`. This enables explicit model management, allowing dynamic control of models via API or HTTP requests.

### 🔄 Load / Unload Models

**Load a model**  
- cURL command:  
  ```bash
  curl -X POST http://localhost:8000/v2/repository/models/yolov5/load

