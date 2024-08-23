import torch
from TTS.api import TTS
import os
import datetime
import random
import string

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
is_gpu = torch.cuda.is_available()

# Set up the model
model_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
tts = TTS(model_name=model_name, gpu=is_gpu)

# List of speakers to test
speakers = ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 
            'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 
            'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 
            'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 
            'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 
            'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 
            'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 
            'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 
            'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 
            'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 
            'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 
            'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']

# Text to be used for all audio generations
text = """ 
Bonjour à tous ! Aujourd'hui, nous plongeons profondément dans le monde complexe des modèles thermodynamiques et climatiques. Ce n'est pas aussi simple que certains voudraient vous le faire croire et la science est très loin d'être établie, comme je vais vous le démontrer. Alors attachez vos ceintures, car nous allons devenir un peu technique !

générique.

Tout d'abord, il n'y a rien de "basique" dans le système climatique terrestre. Si vous avez suivi un cours de dynamique atmosphérique ou ouvert un manuel sur le sujet, vous le sauriez.

Maintenant, examinons un peu, entre guillemets,  la "physiques de base", qui détermine l'ampleur du réchauffement résultant directement de l'augmentation des concentrations de dioxyde de carbone dans l'atmosphère. Préparez-vous, il va y avoir un peu de maths.
"""

text=text.replace('.',';')

def generate_unique_filename(speaker_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"output_{speaker_name}_{timestamp}_{random_string}.wav"

# Ensure the outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Generate audio for each speaker
for speaker in speakers:
    output_filename = generate_unique_filename(speaker.replace(" ", "_"))
    output_path = os.path.join("outputs", output_filename)
    
    print(f"Generating audio for {speaker}...")
    try:
        tts.tts_to_file(text=text, file_path=output_path, speaker=speaker, language='fr')
        print(f"Audio generated successfully for {speaker}. Saved as {output_filename}")
    except Exception as e:
        print(f"Error generating audio for {speaker}: {str(e)}")

print("All speaker tests completed.")