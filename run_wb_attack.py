import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import fire
import os
import numpy as np
import torchvision
import math
import shutil
import pandas as pd
import torch
from huggingface_hub import snapshot_download, hf_hub_download
from utils import processing_utils

@torch.no_grad()
def run_wb_attack(out_parquet_file, parquet_file=None,n_seeds=1,seed_offset=0,outfolder='gen_synthall/',caption_offset=0,n_captions=-1,model='runwayml/stable-diffusion-v1-5',dl_parquet_repo='fraisdufour/sd-stuff',dl_parquet_name='membership_attack_top30k.parquet', compute_images=False, local_dir='.'):
    """
    Runs the whitebox attack against a stable diffusion model run on captions in parquet_file, with a single timestep. Attack score function is the MSE between a one time step generated latent (for LDMs) and the initial noise. By default, we provide a list of 30K captions with the highest whitebox score on SDV1 found at fraisdufour/sd-stuff/membership_attack_top30k.parquet on huggingface, and 2M captions selected by duplication rates (and not by an attack) can be found at fraisdufour/sd-stuff/most_duplicated_metadata.parquet.
    For more details see "A Reproducible Extraction of Training Images from Diffusion Models" https://arxiv.org/abs/2305.08694
    
    Parameters:
        - out_parquet_file (str): Output file path for saving the attack's scores.
        - parquet_file (str, optional): Input parquet containing captions for attack, if it is a local file. Captions should be within field 'caption' in the dataframe.
        - n_seeds (int, optional): Number of random seeds per caption. Default is 1.
        - seed_offset (int, optional): Starting offset for random seed generation. Default is 0.
        - outfolder (str, optional): Output folder path for visualizing attack. Default is disabled.
        - caption_offset (int, optional): Start at a specified offset caption in parquet_file. Useful if you want to divide work amongst several gpus/nodes.
        - n_captions (int, optional): Number of captions to be used from the input parquet. If set to -1, all available captions will be used. Default is -1.
        - model (str, optional): Pre-trained stable diffusion model to be used for generating synthetic data. Default is 'runwayml/stable-diffusion-v1-5'. Use stabilityai/stable-diffusion-2-base for sdv2 models.
        - dl_parquet_repo (str, optional): Huggingface base repo for input parquet.
        - dl_parquet_name (str, optional): Name of parquet within huggingface repo. Default is 'membership_attack_top30k.parquet'.
        - compute_images (bool, optional): Whether to compute and save the generated images (for visualization). Default false.
        - local_dir (str, optional): Local directory path for temporary file storage. Default is '.'.

    Returns:
        None
    
    Saves whitebox scores to out_parquet_file.
    """
    
    if parquet_file is not None:
        d = pd.read_parquet(parquet_file)
    else:
        print(f'downloading from hub {dl_parquet_repo}/{dl_parquet_name}')
        hf_hub_download(repo_id=dl_parquet_repo, filename=dl_parquet_name, repo_type="dataset", local_dir='.')
        d = pd.read_parquet(dl_parquet_name)
    
    # number of diffusion steps. hardcoded to 1 for WB attack.
    steps = 1
    
    from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
    from custom_ksampler_wb_attack import StableDiffusionWBAttack
    pipe = StableDiffusionWBAttack.from_pretrained(model, torch_dtype=torch.float16) 
    pipe.set_scheduler("sample_heun")
    pipe = pipe.to("cuda")

    if n_captions > 0:
        last_caption = min(len(d["caption"]),caption_offset + n_captions)
    else:
        last_caption = len(d["caption"])
    
    d = d.iloc[list(range(caption_offset,last_caption))]
    captions = np.array(d["caption"])[caption_offset:last_caption]
    scores = np.zeros((len(captions),))

    for ci,c in enumerate(captions):
        print(f'synthing caption {ci},{c}...')
        prompt=c
        outfolder_prompt = prompt.replace('/','_')[:min(len(prompt),200)]
        os.makedirs(f'{outfolder}{outfolder_prompt}',exist_ok=True)

        imgs_out = []
        img_group = []
        for seed in range(seed_offset, seed_offset + n_seeds):
            generator = torch.Generator("cuda").manual_seed(seed)
            
            image,z0,latents,score = pipe(prompt,num_inference_steps=1,generator=generator,use_karras_sigmas=True,compute_images=compute_images)
            if compute_images:
                fn = f'{outfolder}{outfolder_prompt}/{seed:04d}.jpg'
                image[0].save(fn)
            
        scores[ci] = score
                                  
    d['scores'] = scores
    d.to_parquet(out_parquet_file)
    
    

if __name__ == '__main__':
    fire.Fire(run_wb_attack)