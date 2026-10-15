# VoiceForge

A small, local voice cloning and text-to-speech app. Upload a short clip of a voice, type some text, and get speech in that voice, with inline sound tags like `[laugh]`. Everything runs on your own PC. No account, no API key, and your audio never leaves your machine.

Built on [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) by Resemble AI (350M parameters, fits in 6-8 GB of VRAM). Inspired by [VoiceStudio](https://github.com/debpalash/VoiceStudio), but written from scratch and much smaller.

## Features

- **Voice cloning** from one clean 10-30 second clip
- **Consistent voice** across long scripts: the same reference clip and a fixed per-voice seed are reused for every sentence chunk
- **Inline tags** such as `[laugh]`, `[chuckle]`, `[cough]`, with buttons that insert them at the cursor
- **Resumable model download**, so a dropped connection doesn't mean starting over
- A voice library, in-browser playback and WAV download

## Requirements

- Windows. Tested on a laptop with an NVIDIA RTX 5050 GPU. Linux and macOS are untested.
- Python **3.11**
- An NVIDIA GPU with 6 GB or more of VRAM. CPU mode works but is slow.

## Setup

Run these in PowerShell. Type only the commands, not the `PS C:\...>` prompt.

```
git clone https://github.com/Rohitftw/voiceforge.git
cd voiceforge
py -3.11 -m venv venv
Set-ExecutionPolicy -Scope Process Bypass
venv\Scripts\Activate.ps1
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**RTX 50-series GPUs (5050, 5060, 5070, ...) need one extra step.** The PyTorch build above does not support them. To check which GPU you have, run:

```
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

If the name starts with RTX 50, or PyTorch prints a warning mentioning `sm_120`, swap the build:

```
pip uninstall -y torch torchaudio
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

pip will print a red message saying chatterbox-tts wants torch 2.6.0. That is expected. Don't re-run `pip install -r requirements.txt` afterwards, or it may swap torch back.

Then download the model and start the app:

```
python download_model.py
python server.py
```

The browser opens at http://127.0.0.1:7860. Next time, only activate the venv and run `python server.py`.

## Usage

1. **Add a voice.** One speaker, no music or background noise, 10-30 seconds. WAV or MP3.
2. **Write your text** and click a tag button to insert a tag at the cursor.
3. **Generate speech**, then play or download the result.

Tip: Chatterbox Turbo's tags cover sounds, not moods. To get different moods, record one clip per mood (calm, excited, sad) and save each as its own voice.

## Limitations

- English only (Turbo model).
- Tags are sounds (`[laugh]`, `[chuckle]`, `[cough]`). There are no `[sad]` or `[excited]` style tags.
- `[sigh]`, `[gasp]` and `[clear throat]` are included but untested. If one is spoken as a word instead of a sound, remove it from the `TAGS` list in `static/index.html`.
- Generation time depends on your GPU and the script length.

## Troubleshooting

| Problem                                                | Fix                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| `No suitable Python runtime found`                     | Install Python 3.11: `winget install -e --id Python.Python.3.11`, then open a new terminal |
| `running scripts is disabled` when activating the venv | Run `Set-ExecutionPolicy -Scope Process Bypass` first                                      |
| `CUDA error: no kernel image is available`             | You have an RTX 50-series GPU. Do the cu128 step in Setup                                  |
| Model fails to load with a 401 or auth error           | Create a free token at huggingface.co/settings/tokens and run `hf auth login`              |
| Model download keeps stopping                          | Run `python download_model.py` again. Progress is kept                                     |

## Responsible use

Only clone voices you have permission to use, such as your own. Do not use this to impersonate people or to mislead others. See the Chatterbox project's own page for its terms.

## Credits and license

- [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI, the model that does the actual cloning and speech

The code in this repository is MIT licensed (see the `LICENSE` file). The model is not included. It is downloaded separately from Hugging Face under its own license.
