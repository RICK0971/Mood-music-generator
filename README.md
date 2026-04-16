# 🎵 Mood-Based Music Generator

Generate music in real-time based on how you're feeling — powered by Meta's MusicGen and a DistilRoBERTa emotion classifier.

## How It Works

1. You type how you're feeling in plain English
2. An NLP model detects your emotion (joy, sadness, anger, fear, etc.)
3. The emotion is mapped to a music prompt
4. Meta's MusicGen generates a unique audio clip to match your mood

## Tech Stack

| Component | Tool |
|---|---|
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` (HuggingFace) |
| Music Generation | `facebook/musicgen-small` via `audiocraft` |
| UI | Gradio |
| Language | Python |

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/mood-music-generator
cd mood-music-generator

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

> **Note:** MusicGen runs best on a GPU. On CPU it will work but be slower (~1-2 min per clip). Use Google Colab (free T4 GPU) for faster results during development.

## Mood → Music Mapping

| Mood | Music Style |
|---|---|
| Joy | Upbeat happy pop, bright melody |
| Sadness | Slow melancholic piano, minor key |
| Anger | Intense heavy rock, aggressive drums |
| Fear | Dark ambient, eerie atmosphere |
| Surprise | Playful quirky, unexpected chords |
| Neutral | Calm lo-fi, relaxing beats |

## Future Ideas

- [ ] Add facial emotion detection via webcam (DeepFace)
- [ ] Let users adjust tempo/genre manually
- [ ] Save and share generated clips
- [ ] Deploy on Hugging Face Spaces

## Deploy on Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Upload `app.py` and `requirements.txt`
4. Set hardware to **T4 GPU** (free tier available)
5. Your app goes live automatically!
