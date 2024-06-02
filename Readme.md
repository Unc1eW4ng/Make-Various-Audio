# MVA:Make Various Audios

first, you need to install the requirements by running the following command:
```bash
pip install -r requirements.txt
```

Then you need to download the pre-trained model from the following link:(The directory organization should also be the same as the following link)

https://github.com/Text-to-Audio/Make-An-Audio


We use gpt-4 to generate different prompts , you need change the api key prompt/forchatgpt.py

```python
api_key = "
```

Then you can run the following command to generate the audio:
```bash
python run.py --caption 'electric guitar' --temperature 0.5
```

The generated audio will be saved in the `results` folder.