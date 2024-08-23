import torch
from TTS.api import TTS
import gradio as gr
import time
from TTS.utils.manage import ModelManager
from TTS.tts.layers.xtts import tokenizer
import os
import datetime
import random
import string

CUSTOM_MODEL_DIR = os.path.join(os.path.expanduser("~"), "tts_models")
os.makedirs(CUSTOM_MODEL_DIR, exist_ok=True)
os.environ["COQUI_TTS_MODEL_DIR"] = CUSTOM_MODEL_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
is_gpu_available = torch.cuda.is_available()

LANGUAGES = ['fr', 'en', 'es', 'it']

# List of available speakers
SPEAKERS = ['none', 'Dionisio Schuyler', 'Baldur Sanjin', 'Kumar Dahl','Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 
            'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 
            'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 
            'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 
            'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 
            'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 
            'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 
            'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 
            'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 
            'Aaron Dreschner', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 
            'Marcos Rudaski']

# List of available models
MODEL_NAMES = [
    'tts_models/multilingual/multi-dataset/xtts_v2',
    'tts_models/multilingual/multi-dataset/xtts_v1.1',
    'tts_models/multilingual/multi-dataset/your_tts',
    'tts_models/multilingual/multi-dataset/bark',
    'tts_models/en/ljspeech/tacotron2-DDC',
    'tts_models/en/ljspeech/tacotron2-DDC_ph',
    'tts_models/en/ljspeech/glow-tts',
    'tts_models/en/ljspeech/speedy-speech',
    'tts_models/en/ljspeech/tacotron2-DCA',
    'tts_models/en/ljspeech/vits',
    'tts_models/en/ljspeech/vits--neon',
    'tts_models/en/ljspeech/fast_pitch',
    'tts_models/en/ljspeech/overflow',
    'tts_models/en/ljspeech/neural_hmm',
    'tts_models/en/vctk/vits',
    'tts_models/en/vctk/fast_pitch',
    'tts_models/en/sam/tacotron-DDC',
    'tts_models/en/blizzard2013/capacitron-t2-c50',
    'tts_models/en/blizzard2013/capacitron-t2-c150_v2',
    'tts_models/en/multi-dataset/tortoise-v2',
    'tts_models/en/jenny/jenny',
    'tts_models/fr/mai/tacotron2-DDC',
    'tts_models/fr/css10/vits'
]

model_manager = ModelManager()

# Get list of reference WAV files
REFERENCES_DIR = 'references'
REFERENCE_FILES = ['none'] + [f for f in os.listdir(REFERENCES_DIR) if f.endswith('.wav')]

def generate_unique_filename(model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    model_short_name = model_name.split('/')[-1]
    return f"output_{model_short_name}_{timestamp}_{random_string}.wav"

def read_text_file(file):
    if file is None:
        return None
    with open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def generate_audio(text, text_file, model_name, speaker, reference_file, temperature, use_gpu, language, progress=gr.Progress()):
    # Check if text is provided directly or via file
    if text_file is not None:
        text = read_text_file(text_file)
    elif not text.strip():
        return None, "Error: No text provided. Please enter text or upload a file."

    # Check if model is downloaded
    model_path, config_path, model_item = model_manager.download_model(model_name)
    if not os.path.isfile(model_path):
        progress(0, desc="Model not found. Downloading...")
        try:
            model_path, config_path, model_item = model_manager.download_model(model_name)
            progress(1, desc="Model downloaded successfully!")
        except Exception as e:
            return None, f"Error downloading model: {str(e)}"
    
    progress(0, desc="Initializing TTS...")
    try:
        tts = TTS(model_name=model_name, gpu=use_gpu)
    except Exception as e:
        return None, f"Error initializing TTS model: {str(e)}"

    output_filename = generate_unique_filename(model_name)
    output_path = os.path.join("outputs", output_filename)
    
    # Ensure the outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    text = text.replace('.', ';')
    text = text.replace(',', ';')

    progress(0.1, desc="Performing TTS... (percentage is not accurate)")

    # Actual TTS generation
    try:
        if 'your_tts' in model_name or 'multilingual' in model_name:
            if reference_file != 'none':
                speaker_wav = os.path.join(REFERENCES_DIR, reference_file)
                tts.tts_to_file(text=text, file_path=output_path, speaker_wav=speaker_wav, language=language, split_sentences=True, temperature=temperature, repetition_penalty=10.0,length_penalty=2.0, top_k=50, top_p=0.85)
            elif speaker is not None and speaker != 'none':
                tts.tts_to_file(text=text, file_path=output_path, speaker=speaker, language=language, split_sentences=True, temperature=temperature, repetition_penalty=10.0,length_penalty=2.0, top_k=50, top_p=0.85)
            else:
                tts.tts_to_file(text=text, file_path=output_path, language=language, split_sentences=True, temperature=temperature, repetition_penalty=10.0,length_penalty=1.0, top_k=50, top_p=0.85)
        else:
            tts.tts_to_file(text=text, file_path=output_path, split_sentences=True, temperature=temperature, repetition_penalty=10.0,length_penalty=1.0, top_k=50, top_p=0.85)
        progress(0.95, desc="Done performing TTS.")
        return output_path, f"Audio generated successfully! Saved as {output_filename}"
    except Exception as e:
        progress(1.0, desc="Error.")
        return None, f"Error generating audio: {str(e)}"
    
    

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Speech coqui-ai TTS FRENCH")
    gr.Markdown("Generate audio from text or uploaded .txt file using a wide range of TTS models. Each output file will have a unique name.")
    
    with gr.Row():
        text_input = gr.Text(
            label="Text Input",
            value="""Ceci est le début.
                Bonjour à tous, je suis vraiment heureux de vous parler aujourd'hui. Ce texte est spécialement conçu pour évaluer et améliorer les capacités de synthèse vocale en français. Nous allons explorer une grande diversité de phrases pour bien saisir les nuances et les subtilités de la langue française.
                Chaque jour commence par un lever de soleil magnifique, où la lumière dorée illumine le paysage. Les oiseaux chantent joyeusement, créant une atmosphère paisible. Dans notre quotidien, il est important de prendre un moment pour apprécier ces petits plaisirs. Le matin, je prends toujours une tasse de café bien chaud pour me réveiller et me préparer à affronter la journée. Le café est un rituel pour beaucoup d'entre nous. Un moment de calme avant de plonger dans les activités du jour.
                Ceci est la fin.""",
            placeholder="Enter text here or upload a .txt file below"
        )
        text_file = gr.File(label="Upload Text File (.txt)", file_types=[".txt"])
    
    with gr.Row():
        model_dropdown = gr.Dropdown(label="Model", choices=MODEL_NAMES, value=MODEL_NAMES[0])
        speaker_dropdown = gr.Dropdown(label="Speaker", value='none', choices=SPEAKERS, visible=True)
    
    with gr.Row():
        reference_dropdown = gr.Dropdown(label="Reference Speaker WAV", choices=REFERENCE_FILES, value=REFERENCE_FILES[4])
        temperature_slider = gr.Slider(minimum=0.1, maximum=1.9, value=0.1, step=0.1, label="Temperature")

    with gr.Row():
        language_dropdown = gr.Dropdown(label="Language", choices=LANGUAGES, value='fr')
        use_gpu_checkbox = gr.Checkbox(label="Use GPU (if available, otherwise use CPU, slower)", value=is_gpu_available)
    
    with gr.Row():
        generate_button = gr.Button("Generate Audio")
        abort_button = gr.Button("Abort Generation")
    
    audio_output = gr.Audio(label="Generated Audio")
    status_output = gr.Textbox(label="Status")
    
    job = generate_button.click(
        generate_audio,
        inputs=[text_input, text_file, model_dropdown, speaker_dropdown, reference_dropdown, temperature_slider, use_gpu_checkbox, language_dropdown],
        outputs=[audio_output, status_output]
    )
    
    abort_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[job]
    )

demo.launch()