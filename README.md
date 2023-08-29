# Llama 2 MPS

This branch represents a fork of [LLAMA2](https://github.com/facebookresearch/llama) with GPU acceleration for M1/M2 macs.  

The model weights are distributed in bfloat16 format, which is at this time not supported by MPS, and pytorch loads the weights in full float32 before using the model in float16, which means it will use up to 2x of the model size in RAM at loading stage. If you have your weights converted to float16, you can try using `--weights_in_float16=True` argument.

To run the example:
```bash
python3.10 -m torch.distributed.launch --use_env example_text_completion.py --ckpt_dir PATH_TO_MODEL_WEIGHTS_v2_7B/ --tokenizer_path .../tokenizer.model --max_batch_size=1
```

This fork has minimal changes to the original repo. If you are looking for a more memory-efficient way to run this, there are a few options: transformers library, tinygrad, llama.cpp.