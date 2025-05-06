import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU0 Used: {info.used / (1024 ** 2):.2f} MB")
pynvml.nvmlShutdown()