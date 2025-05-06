# Triton-Inference-Server

## 🧠 Triton Model Management


To manually load or unload models during runtime, make sure the Triton Inference Server is launched with `--model-control-mode=explicit`. This enables explicit model management, allowing dynamic control of models via API or HTTP requests.

### 🔄 Load Models 
- cURL command:  
  ```bash
  curl -X POST http://localhost:8000/v2/repository/models/yolov5/load
- Python API:  
  ```python
  client.load_model("yolov5")

 ### ❌ Unload Models 
- cURL command:  
  ```bash
  curl -X POST http://localhost:8000/v2/repository/models/yolov5/unload
- Python API:  
  ```python
  client.unload_model("yolov5")

ℹ️ Replace "yolov5" with actual model name. Make sure the model exists in the configured `--model-repository` path.

