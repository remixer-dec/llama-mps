# LLaMa MPS fork

This is a fork of https://github.com/markasoftware/llama-cpu which is a fork of https://github.com/facebookresearch/llama. The goal of this fork is to use MPS acceleration on Apple M1/M2 devices.   
  
Please check the original repos for installation instructions. After you're done, run this
 `torchrun example.py --ckpt_dir ../7B  --tokenizer_path ../tokenizer.model --max_batch_size=1` with correct paths to the models. You might need to set up env. variable PYTORCH_ENABLE_MPS_FALLBACK=1  
   
This fork is experimental, currently at the stage which allows to run the model with MPS, it might produce different results than the actual model, if you know how to improve something, send a PR.  

After the model is loaded, inference for 20 tokens/words takes about 3 seconds on a 24-core M1 Max vs 12+ minutes on a CPU (running on a single core). It always goes above 32gb of RAM, writing 2-4gb to ssd on every launch, but consumes less memory after it is loaded.  
  
You might notice, that the output of the model often has -1 in tokens and empty/repetitive text. Similar behavior is mentioned [in this guide](https://rentry.org/llama-tard#im-getting-shitty-results), so I am not sure if it is related to issues in the port, try changing the seed and output length to get different results. Any ideas on how to improve it are welcome.
