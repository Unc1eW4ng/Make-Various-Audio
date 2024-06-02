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

if __name__ == '__main__':
    with torch.no_grad():
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt('/root/autodl-tmp/CLAP/630k-audioset-best.pt') # download the default pretrained checkpoint.
        eval_pairs = [[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        with open('data/test.csv', 'r') as f:
            csv = pd.read_csv(f)

        inner_sim = 0
        cross_sim = 0
        for i in tqdm(range(100)):
            try:
                caption = csv['source_captions'][i]
                newcaptions = get_various_caption(caption)
                oldvec = model.get_text_embedding([caption,'occupy'],use_tensor=True)[0]
                newvecs = model.get_text_embedding(newcaptions,use_tensor=True)
                for j in range(5):
                    csv.loc[i, f'new{j}']=newcaptions[j]
                    inner_sim += torch.nn.functional.cosine_similarity(oldvec.unsqueeze(0),newvecs[j].unsqueeze(0))
                for pair in eval_pairs:
                    cross_sim += torch.nn.functional.cosine_similarity(newvecs[pair[0]].unsqueeze(0),newvecs[pair[1]].unsqueeze(0))
            except Exception as e:
                print("An error occurred:", e)

        print('inner',inner_sim/500)
        print('cos',cross_sim/1000)

        with open(f'data/{getmodel()}.csv', 'w') as f:
            csv.to_csv(f, index=False)

