import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

TEXT="""Bonjour à tous ! Aujourd'hui, nous plongeons profondément dans le monde complexe des modèles thermodynamiques et climatiques. Ce n'est pas aussi simple que certains voudraient vous le faire croire et la science est très loin d'être établie, comme je vais vous le démontrer. Alors attachez vos ceintures, car nous allons devenir un peu technique !
générique.
Tout d'abord, il n'y a rien de "basique" dans le système climatique terrestre. Si vous avez suivi un cours de dynamique atmosphérique ou ouvert un manuel sur le sujet, vous le sauriez.
Maintenant, examinons un peu la, entre guillemets, "physiques de base", qui détermine l'ampleur du réchauffement résultant directement de l'augmentation des concentrations de dioxyde de carbone dans l'atmosphère. Préparez-vous, il va y avoir un peu de maths !
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