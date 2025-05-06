import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")  # http

# test model name
model_name = "yolov5"

# load model
client.load_model(model_name)
print(f"model {model_name} load ")

# unload model
client.unload_model(model_name)
print(f"model {model_name} unload")

# check status (get_model_metadata / get_model_config)
if client.is_model_ready(model_name):
    print(f"model {model_name} is ready")
else:
    print(f"model {model_name} is not ready")
