import torch
import numpy as np
import pandas as pd
from MAA.vocoder.bigvgan.models import VocoderBigVGAN
from MAA.ldm.models.diffusion.ddim import DDIMSampler
from MAA.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import soundfile
from tqdm import tqdm
import os
device = 'cuda' # change to 'cpu‘ if you do not have gpu. generating with cpu is very slow.
SAMPLE_RATE = 16000



def initialize_model(config, ckpt,device=device):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt,map_location='cpu')["state_dict"], strict=False)

    model = model.to(device)
    model.cond_stage_model.to(model.device)
    model.cond_stage_model.device = model.device
    print(model.device,device,model.cond_stage_model.device)
    sampler = DDIMSampler(model)

    return sampler

def dur_to_size(duration):
    latent_width = int(duration * 7.8)
    if latent_width % 4 != 0:
        latent_width = (latent_width // 4 + 1) * 4
    return latent_width

def gen_wav(sampler,vocoder,prompt,ddim_steps,scale,duration,n_samples):
    latent_width = dur_to_size(duration)
    start_code = torch.randn(n_samples, sampler.model.first_stage_model.embed_dim, 10, latent_width).to(device=device, dtype=torch.float32)
    
    uc = None
    if scale != 1.0:
        uc = sampler.model.get_learned_conditioning(n_samples * [""])
    c = sampler.model.get_learned_conditioning(n_samples * [prompt])
    shape = [sampler.model.first_stage_model.embed_dim, 10, latent_width]  # 10 is latent height 
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                        conditioning=c,
                                        batch_size=n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        x_T=start_code)

    x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)

    wav_list = []
    for idx,spec in enumerate(x_samples_ddim):
        wav = vocoder.vocode(spec)
        if len(wav) < SAMPLE_RATE * duration:
            wav = np.pad(wav,SAMPLE_RATE*duration-len(wav),mode='constant',constant_values=0)
        wav_list.append(wav)
    return wav_list


def Gen_wavs(captions,initial_cap):
    wavs=[]
    sampler = initialize_model('MAA/configs/text_to_audio/txt2audio_args.yaml', 'MAA/useful_ckpts/maa1_full.ckpt')
    vocoder = VocoderBigVGAN('MAA/useful_ckpts/bigvgan',device=device)
    initial_cap = initial_cap.replace(' ','_')

    if os.path.exists(f'results/{initial_cap}') == False:
        os.mkdir(f'results/{initial_cap}')
    for i in range(len(captions)):
        captions[i] = captions[i].replace(' ','_')
        wav = gen_wav(sampler,vocoder,prompt=captions[i],ddim_steps=100,scale=3,duration=10,n_samples=1)[0]
        soundfile.write(f'results/{initial_cap}/{captions[i]}.wav',wav,samplerate=SAMPLE_RATE)  
        wavs.append(f'results/{initial_cap}/{captions[i]}.wav')
    return wavs

    

# model = 'gpt-4'
# csv = pd.read_csv(f'data/{model}.csv')
# if os.path.exists(f'results/{model}') == False:
#     os.mkdir(f'results/{model}')
# for i in tqdm(range(len(csv))):
#     for j in range(5):
#         prompt = csv[f'new{j}'][i]
#         try:
#             if len(prompt) <5 :
#                 continue
#         except:
#             continue

        

