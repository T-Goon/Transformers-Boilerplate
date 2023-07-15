# Transformers Boilerplate


## Requirements

- Windows 10/11
- [WSL2 environment](https://learn.microsoft.com/en-us/windows/wsl/install)
- (Optinal) [Pyenv](https://github.com/pyenv/pyenv) for Python version management
- Python 3.10
- Pipenv `pip install pipenv`
- (Optional) [VS Code with Remote Explorer](https://code.visualstudio.com/docs/remote/wsl) for Developing on WSL
- [Cuda 11.7](https://developer.nvidia.com/cuda-11-7-1-download-archive)
- [Cuda capable GPU](https://developer.nvidia.com/cuda-gpus)
- (Optional) [MSI Afterburner](https://www.msi.com/Landing/afterburner/graphics-cards) to track hardware utilization

## Setup
[This document](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-wsl) may also be helpful durring setup.

0. Ensure that your graphics drivers are installed and up to date.
1. Set up your WSL2 environment
2. Install python onto WSL
3. Install CUDA 11.7 onto WSL
    - After installing CUDA I needed to add some directories to the PATH in WSL.
    - You can add these to your .bashrc or some other init script for the terminal you use:
        - `export PATH="/usr/local/cuda/bin:$PATH"`
        - `export PATH="/usr/local/cuda-11.7/targets/x86_64-linux/lib:$PATH"`
4. Clone this repo somewhere in WSL
5. To ensure compatibility, install packages from Pipfile.lock `pipenv sync`

## Files
- `inference.py`: Load a model from Huggingface, apply 4 bit quantization, and run inference
- `peft-ft.py`: Load a model from Huggingface, apply 4 bit quantization, fine-tune using QLORA, save the model, and do inference once
- `peft-infer.py`: Load a fine-tunned peft model and do inference.

# Running
`pipenv run python <file_name>.py`