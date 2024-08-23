# Text-to-Speech (TTS) Application using Python TTS. French (by default, can be changed)

This application uses the coqui-ai TTS library to generate speech from text input. It supports multiple languages and models, and provides a user-friendly interface using Gradio.

## Setup Instructions

Follow these steps to set up the environment and run the application:

### 1. Install Miniconda

Miniconda is a minimal installer for conda. If you don't have it installed, follow these steps:

1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Choose the appropriate version for your operating system
3. Run the installer and follow the prompts

### 2. Create a Conda Environment

Once Miniconda is installed, create a new conda environment:

```bash
conda create -n tts python=3.11
```

Activate the environment:

```bash
conda activate tts
```

### 3. Install Required Packages

Install the TTS and Gradio packages using pip:

```bash
pip install tts==0.22.0 gradio==4.42.0
```

### 4. Run the Application

1. Ensure you're in the directory containing the `main.py` file
2. Run the following command:

```bash
gradio main.py
```

3. Open the URL provided in the console output in your web browser

## Usage

1. Enter the text you want to convert to speech in the "Text Input" box, or upload a .txt file
2. Select a TTS model from the dropdown menu
3. Choose a speaker (if applicable)
4. Select a reference speaker WAV file (if desired). You can put new references inside the `references` folder.
5. Adjust the temperature using the slider
6. Check the "Use GPU" box if you want to use GPU acceleration (if available)
7. Select the desired language from the dropdown menu
8. Click the "Generate Audio" button, it generates playable audio in the GUI, and files are saved in the `outputs` folder.
9. Listen to the generated audio or download it

## Troubleshooting

- If you encounter any issues with GPU acceleration, try unchecking the "Use GPU" box
- Ensure you have the latest graphics drivers installed if using GPU acceleration
- If you get any package-related errors, try updating the packages:
  ```bash
  pip install --upgrade tts gradio
  ```