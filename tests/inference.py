import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

TEXT="""
Ceci est le début.
Bonjour à tous, je suis vraiment heureux de vous parler aujourd'hui. Ce texte est spécialement conçu pour évaluer et améliorer les capacités de synthèse vocale en français. Nous allons explorer une grande diversité de phrases pour bien saisir les nuances et les subtilités de la langue française.
Chaque jour commence par un lever de soleil magnifique, où la lumière dorée illumine le paysage. Les oiseaux chantent joyeusement, créant une atmosphère paisible. Dans notre quotidien, il est important de prendre un moment pour apprécier ces petits plaisirs. Le matin, je prends toujours une tasse de café bien chaud pour me réveiller et me préparer à affronter la journée. Le café est un rituel pour beaucoup d'entre nous. Un moment de calme avant de plonger dans les activités du jour.
Ceci est la fin.
"""

print("Loading model...")
config = XttsConfig()
config.load_json("./XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference2.wav"])

print("Inference...")
out = model.inference(
    TEXT,
    "fr",
    gpt_cond_latent,
    speaker_embedding,
    enable_text_splitting=True,
    length_penalty=2.0,
    repetition_penalty=20.0,
    top_k=1,
    top_p=10.00,
    temperature=0.01, # Add custom parameters here
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)