import numpy as np
import cv2
import time
from datetime import datetime
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import pynvml

def preprocess_frame(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

def show_gpu_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU Memory Usage:")
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_mb = mem_info.used / (1024 ** 2)
        total_mb = mem_info.total / (1024 ** 2)
        print(f"GPU {i}: Used {used_mb:.2f} MB / {total_mb:.2f} MB")
    pynvml.nvmlShutdown()

def do_inference(client, model_name, input_data, output_names):
    inputs = [InferInput("images", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [InferRequestedOutput(name) for name in output_names]
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return {name: response.as_numpy(name) for name in output_names}

def load_model(client, model_name):
    try:
        client.load_model(model_name)
        print(f"model {model_name} load")
    except Exception as e:
        print(f"load {model_name} failed：{e}")

def unload_model(client, model_name):
    try:
        client.unload_model(model_name)
        print(f"model {model_name} unload")
    except Exception as e:
        print(f"unload {model_name} failed：{e}")

if __name__ == "__main__":
    video_path = "MOT20-02.mp4"
    client = InferenceServerClient(url="localhost:8000")
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    current_model = None  # Track which model is currently loaded

    print("Start processing video， GPU memory trace + dynamic model control (Ctrl+C to stop)\n")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            input_data = preprocess_frame(frame)

            print(f"\nFrame {frame_id} - before infer GPU Memory")
            show_gpu_memory()

            batch_index = (frame_id - 1) // 10
            if batch_index % 2 == 0:
                # Use yolo_ensemble for this 10-frame batch
                if current_model != "yolo_ensemble":
                    if current_model:
                        unload_model(client, current_model)
                    load_model(client, "yolo_ensemble")
                    current_model = "yolo_ensemble"
                _ = do_inference(client, "yolo_ensemble", input_data, ["output_yolov8n", "output_yolo11n"])
            else:
                # Use yolo_ensemble2 for this 10-frame batch
                if current_model != "yolo_ensemble2":
                    if current_model:
                        unload_model(client, current_model)
                    load_model(client, "yolo_ensemble2")
                    current_model = "yolo_ensemble2"
                _ = do_inference(client, "yolo_ensemble2", input_data, ["output_yolov5", "output_yolov8n"])

            print(f"Frame {frame_id} - after infer GPU Memory")
            show_gpu_memory()

    except KeyboardInterrupt:
        print("\nstop inference by user")

    finally:
        cap.release()
        if current_model:
            unload_model(client, current_model)
        print("stop video preprocessing")
