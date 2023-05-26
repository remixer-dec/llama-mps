# LLaMa MPS fork (multimodal llama-adapter branch)  
This is a fork of [LLaMA-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter/) which is a fork of [llama](https://github.com/facebookresearch/llama). The goal of this fork is to use GPU acceleration on Apple M1/M2 devices. This branch provides support for [LLaMA-Adapter with image input support (multimodal version)](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal).  

## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n llama_adapter_v2 python=3.10 -y
  pip install -r requirements.txt
  ```
* set the path to the llama model in the file that you want to run  

run `python demo.py` for the demo and `python gradio_app.py` for interactive web-server and API.  
for more information, check out the original repo