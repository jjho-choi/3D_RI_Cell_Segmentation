# 3D RI Cell Segmentation
This repository contains PyTorch implementation of [3D RI Cell Segmentation](https://www.biorxiv.org/content/10.1101/2021.05.23.445351v1.abstract).

## System requirements and installation guide
Installation should be fairly quick (typically less than an hour). On a computer with CUDA-compatible GPUs and Linux operating system (e.g., Ubuntu 16.04), install CUDA/cuDNN and Python 3 (tested on 3.7) with the following packages:
```yaml
pytorch >= 1.0
numpy
scipy
cc3d
h5py
tqdm
```

## Demo and instructions for use
### YAML script
class parses arguments from the `yaml_file`. 

Exmaple of `yaml_file`:

```yaml
# scripts/test.yaml
preprocess:
  zoomed_size: 256
  patch_size: 128
  z_size: 64
  center_crop_size: 90
  cell_resize_size: 128

model:
  suborgan: './weights/sub_organ.pth'
  cell_by_cell: './weights/cell_inst.jit'

data:
  path: './data/20200609.153947.886.Hep G2-094.TCF'

save:
  path: './save'

```
### Inference
Download sample data and weight file from this [link](https://drive.google.com/drive/folders/1j8qjk3tZL4Pk32855jwpww9A9_XygcRa?usp=sharing). 

Then, run the python script with the following command as bellow.
```bash
➜ python3 run.py config --configs/test.yaml
```
The demo data in the 'data' folder will output segmentation masks inferred from the input RI tomograms. You can identify results in 'show_results.ipynb' file. In order to run our code with your own data, organize your RI tomogram in this format and repeat the procedures above. Run time depends on data size and hardware; for a full-sized tomogram, it is expected to take less than a minute with a NVIDIA V100 GPU. 
