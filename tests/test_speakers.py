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
Ceci est le début.
Bonjour à tous, je suis vraiment heureux de vous parler aujourd'hui. Ce texte est spécialement conçu pour évaluer et améliorer les capacités de synthèse vocale en français. Nous allons explorer une grande diversité de phrases pour bien saisir les nuances et les subtilités de la langue française.
Chaque jour commence par un lever de soleil magnifique, où la lumière dorée illumine le paysage. Les oiseaux chantent joyeusement, créant une atmosphère paisible. Dans notre quotidien, il est important de prendre un moment pour apprécier ces petits plaisirs. Le matin, je prends toujours une tasse de café bien chaud pour me réveiller et me préparer à affronter la journée. Le café est un rituel pour beaucoup d'entre nous. Un moment de calme avant de plonger dans les activités du jour.
Ceci est la fin.
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