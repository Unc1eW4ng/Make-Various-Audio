from prompt.forchatgpt import get_various_caption,getmodel
from Recommend import rank
import pandas as pd
from tqdm import tqdm
from AudioGen import Gen_wavs
import numpy as np
import librosa
import torch
import laion_clap
import argparse

parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('--caption', type=str, required=True,
                    help='Caption string, must not be empty')

parser.add_argument('--temperature', type=float, default=0.5,
                    help='Temperature value, default is 0.5')

args = parser.parse_args()

print(f'Caption: {args.caption}')
print(f'Temperature: {args.temperature}')

print('Getting various captions...')
while True:
    try:
        captions = get_various_caption(args.caption)
        break
    except Exception as e:
        print(f'An error occurred: {e}')
        print('Retrying...')
print('Various captions obtained.')

for i, caption in enumerate(captions):
    print(f'Caption {i}: {caption}')
print(captions)

print('Generating wavs...')
wavs = Gen_wavs(captions, args.caption)
print('Wavs generated.')

print('Ranking wavs...')
ranked_wavs = rank(wavs, args.caption, args.temperature)
print('Wavs ranked.')

print('Ranked wavs:')
for i, wav in enumerate(ranked_wavs):
    print(f'Rank {i}: {wav}')