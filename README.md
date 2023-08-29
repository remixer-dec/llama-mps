# LLaMa MPS fork

This is a fork of https://github.com/markasoftware/llama-cpu which is a fork of https://github.com/facebookresearch/llama. The goal of this fork is to use GPU acceleration on Apple M1/M2 devices.   

[LLaMa-adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter) support has been added in a separate [branch](https://github.com/remixer-dec/llama-mps/tree/adapter-model)!  
[Multi-modal LLaMa-adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal) support has been added in a separate [branch](https://github.com/remixer-dec/llama-mps/tree/multimodal-adapter)!  
**Llama v2** support has been added in a separate [branch](https://github.com/remixer-dec/llama-mps/tree/llama2)  
  
Please check the original repos for installation instructions. After you're done, run this
 `torchrun example.py --ckpt_dir ../7B  --tokenizer_path ../tokenizer.model --max_batch_size=1` with correct paths to the models. You might need to set up env. variable PYTORCH_ENABLE_MPS_FALLBACK=1  
   
This fork is experimental, currently at the stage which allows to run a full non-quantized model with MPS.  

After the model is loaded, inference for max_gen_len=20 takes about 3 seconds on a 24-core M1 Max vs 12+ minutes on a CPU (running on a single core). For 7B model, it always goes above 32gb of RAM, writing 2-4gb to ssd (swap) on every launch, but consumes less memory after it is loaded.  
  
If you notice, that the output of the model has empty/repetitive text, try using a fresh version of python/pytorch. For me it was giving bad outputs with Python 3.8.15 and pytorch 1.12.1. After trying it with python3.10 and torch 2.1.0.dev20230309 the model worked as expected and produced high-quality outputs.
