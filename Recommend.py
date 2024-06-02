from prompt.forchatgpt import get_various_caption,getmodel
import pandas as pd
from tqdm import tqdm


import numpy as np
import librosa
import torch
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def rank(audios,prompt,temperature=0.5):
    with torch.no_grad():
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt('/root/autodl-tmp/CLAP/630k-audioset-best.pt') # download the default pretrained checkpoint.
        text_embed = model.get_text_embedding([prompt,'occupy'],use_tensor=True)[0]
        audio_embed = model.get_audio_embedding_from_filelist(x = audios, use_tensor=True)
        print(audio_embed.shape)
        inner_sim = []
        tmp_sim = {}
        cross_sim = []
        for i in range(len(audios)):
            inner_sim.append(torch.nn.functional.cosine_similarity(text_embed.unsqueeze(0),audio_embed[i].unsqueeze(0)))
        for i in range(len(audios)):
            for j in range(i+1,len(audios)):
                tmp_sim[(i,j)] = torch.nn.functional.cosine_similarity(audio_embed[i].unsqueeze(0),audio_embed[j].unsqueeze(0))
        for i in range(len(audios)):
            sum = 0
            for j in range(i,len(audios)):
                if i != j:
                    sum += tmp_sim[(i,j)]
            cross_sim.append(1-sum/len(audios))
        
        grade = []
        for i in range(len(audios)):
            grade.append((1-temperature)*inner_sim[i]+temperature*cross_sim[i])

        #sort audios by grade
        audios = [x for _,x in sorted(zip(grade,audios),reverse=True)]
        return audios
        
if __name__ == '__main__':
    audios = ['results/gpt-4/0_0.wav','results/gpt-4/0_1.wav','results/gpt-4/0_2.wav','results/gpt-4/0_3.wav','results/gpt-4/0_4.wav']
    prompt = 'guitar  piano  and simple percussion'
    ranked_audios = rank(audios,prompt)
    print(ranked_audios)
