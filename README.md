# Triton-Inference-Server
## 🧰 Triton Server & Client Setup 
This section documents how to start the Triton Inference Server, verify its status, and make inference requests from the client container.

### [1] Start the Triton Server

- Basic server launch with model repository:
  ```bash
  docker run --gpus=1 -p8000:8000 -p8001:8001 -p8002:8002 \
    -v /home/os-iris.ys.liu/test_triton/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.05-py3 \
    tritonserver --model-repository=/models
- Add shared memory for large model usage:
  ```bash
  docker run --gpus all --shm-size=1g --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /home/os-iris.ys.liu/test_triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.05-py3 \
  tritonserver --model-repository=/models
- Install model dependencies (e.g. OpenCV, PyTorch) and enable explicit model control:
  ```bash
  docker run --gpus all --shm-size=1g --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /home/os-iris.ys.liu/test_triton/model_repository:/models \
  --entrypoint /bin/bash nvcr.io/nvidia/tritonserver:23.05-py3 \
  -c "pip install --no-cache-dir opencv-python-headless torch torchvision && \
      tritonserver --model-repository=/models --model-control-mode=explicit"

### [2] Confirm Triton Server is Running
- Use `curl` to check server readiness:
  ```bash
  curl -v localhost:8000/v2/health/ready

### [3] Send Inference Request from Client Container
- Start the client container (SDK image):
  ```bash
  docker run -it --net=host nvcr.io/nvidia/tritonserver:23.05-py3-sdk

### [4] Test Inference Using Built-in Image Client
- Inside the client container, run the example image inference client:
  ```bash
  install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
  
📝 Docker Notes
`--rm`: Automatically removes the container when it exits.
`-v`: Mounts a local host path into the container.


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

- Python Example:  
  ```python
  get_model_repository_index()

