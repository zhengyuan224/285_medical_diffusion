# Medical Diffusion

Code mainly from "Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Synthesis"
(see https://arxiv.org/abs/2211.03364).

Do some modifivations to the original code so that it can fit our dataset.

![Generated Samples by our Medical Diffusion model](assets/generated_samples.gif)

# System Requirements
This code has been tested on Ubuntu 20.04 and an NVIDIA RTX 3090 GPU. Furthermore it was developed using Python v3.8.

# Setup
In order to run our model, we suggest you create a virtual environment 
```
conda create -n medicaldiffusion python=3.8
``` 
and activate it with 
```
conda activate medicaldiffusion
```
Subsequently, download and install the required libraries by running 
```
pip install -r requirements.txt
```

# Training

vqgan training

```
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

```
PYTHONPATH=. python train/train_vqgan.py \
  dataset=default \
  dataset.root_dir=/root/medicaldiffusion/my_dataset \
  model=vq_gan_3d \
  model.gpus=1 \
  model.batch_size=1 \
  model.num_workers=4 \
  model.n_hiddens=32 \
  model.embedding_dim=16 \
  model.n_codes=1024 \
  model.max_epochs=100 \
  model.precision=16 \
  model.accumulate_grad_batches=4
```

ddpm training

```
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

```
PYTHONPATH=. python train/train_ddpm.py \
  model=ddpm \
  dataset=default \
  dataset.root_dir=/root/medicaldiffusion/my_dataset \
  model.results_folder_postfix='own_dataset' \
  model.vqgan_ckpt=/root/medicaldiffusion/checkpoints/DEFAULT/lightning_logs/version_1/checkpoints/latest_checkpoint.ckpt \
  model.diffusion_img_size=64 \
  model.diffusion_depth_size=8 \
  model.diffusion_num_channels=8 \
  model.dim_mults=[1,2,4,8] \
  model.batch_size=2 \
  model.save_and_sample_every=50 \
  model.gpus=0
```

# Citation
code from
```
@misc{https://doi.org/10.48550/arxiv.2211.03364,
  doi = {10.48550/ARXIV.2211.03364},
  url = {https://arxiv.org/abs/2211.03364},
  author = {Khader, Firas and Mueller-Franzes, Gustav and Arasteh, Soroosh Tayebi and Han, Tianyu and Haarburger, Christoph and Schulze-Hagen, Maximilian and Schad, Philipp and Engelhardt, Sandy and Baessler, Bettina and Foersch, Sebastian and Stegmaier, Johannes and Kuhl, Christiane and Nebelung, Sven and Kather, Jakob Nikolas and Truhn, Daniel},
  title = {Medical Diffusion - Denoising Diffusion Probabilistic Models for 3D Medical Image Generation},
  publisher = {arXiv},
  year = {2022},
}
```


# Acknowledgement
This code is heavily build on the following repositories:

(1) https://github.com/SongweiGe/TATS

(2) https://github.com/lucidrains/denoising-diffusion-pytorch

(3) https://github.com/lucidrains/video-diffusion-pytorch
