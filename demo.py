import cv2
import llama
import torch
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/"

model, preprocess = llama.load("BIAS-7B", llama_dir, device)

prompt = llama.format_prompt('How many cats can you find in this image?')
img = Image.fromarray(cv2.imread("cat.jpg"))
img = preprocess(img).unsqueeze(0).half().to(device)

result = model.generate(img, [prompt])[0]

print(result)

prompt = llama.format_prompt('Describe this image.')
result = model.generate(img, [prompt])[0]

print(result)