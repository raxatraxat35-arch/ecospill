import torch

print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Видеокарта:", torch.cuda.get_device_name(0))