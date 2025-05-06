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

## 📚 Querying Triton Model Repository Index
You can query the current model repository index to check which models are available and their status.
### 🔧 Command-line Version (Outside the Container)
Use the following `curl` command directly from your host machine:  
- cURL command:  
  ```bash
  curl -X POST localhost:8000/v2/repository/index

### 💻 Programmatic Version (Inside Client-side Container)
- C++ Example:  
  ```bash
  # Move and run the test script (optional helper)
  $ mv qa/L0_sdk/test.sh .
  $ bash test.sh .

  # Compile the client example
  $ g++ triton_client_q.c -o triton_client_q \
    -I/workspace/triton_client/include \
    -L/workspace/triton_client/lib -lhttpclient
  
  # Run the client binary
  $ ./triton_client_q

- Python API:  
  ```python
  get_model_repository_index()

