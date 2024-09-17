import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Device being used: {'cuda' if torch.cuda.is_available() else 'cpu'}")