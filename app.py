import gradio as gr
from transformers import pipeline
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
import tempfile
import os

# ----------------------------
# 1. Load Models
# ----------------------------

# Mood detection from text
sentiment_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# MusicGen (use 'small' to run on CPU; switch to 'medium' if you have GPU)
music_model = MusicGen.get_pretrained("facebook/musicgen-small")
music_model.set_generation_params(duration=10)  # seconds


# ----------------------------
# 2. Mood → Music Prompt Mapping
# ----------------------------

MOOD_TO_PROMPT = {
    "joy":      "upbeat happy pop music, bright melody, energetic, feel-good",
    "sadness":  "slow melancholic piano, emotional, minor key, soft and reflective",
    "anger":    "intense heavy rock, fast tempo, aggressive drums, electric guitar",
    "fear":     "dark ambient, tense strings, eerie atmosphere, suspenseful",
    "surprise": "playful quirky music, unexpected chord changes, fun and whimsical",
    "disgust":  "dissonant experimental music, unsettling tones, avant-garde",
    "neutral":  "calm lo-fi music, gentle beats, relaxing, background study music",
}


# ----------------------------
# 3. Core Functions
# ----------------------------

def detect_mood(user_input: str) -> str:
    """Detect mood from user text input."""
    result = sentiment_model(user_input)[0][0]
    label = result["label"].lower()
    return label


def generate_music(mood: str) -> str:
    """Generate music based on mood and return path to audio file."""
    prompt = MOOD_TO_PROMPT.get(mood, MOOD_TO_PROMPT["neutral"])

    # Generate
    wav = music_model.generate([prompt])  # shape: [1, channels, samples]

    # Save to temp file
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "output")
    audio_write(
        output_path,
        wav[0].cpu(),
        music_model.sample_rate,
        strategy="loudness"
    )
    return output_path + ".wav"


def run_pipeline(user_input: str):
    """Full pipeline: text → mood → music."""
    if not user_input.strip():
        return "Please enter how you're feeling.", None

    mood = detect_mood(user_input)
    friendly_mood = mood.capitalize()
    prompt_used = MOOD_TO_PROMPT.get(mood, MOOD_TO_PROMPT["neutral"])

    audio_path = generate_music(mood)

    status = f"**Detected Mood:** {friendly_mood}\n\n**Music Prompt Used:** _{prompt_used}_"
    return status, audio_path


# ----------------------------
# 4. Gradio UI
# ----------------------------

with gr.Blocks(title="Mood Music Generator") as demo:
    gr.Markdown("# 🎵 Mood-Based Music Generator")
    gr.Markdown("Tell me how you're feeling and I'll generate music to match your vibe.")

    with gr.Row():
        text_input = gr.Textbox(
            label="How are you feeling right now?",
            placeholder="e.g. I'm feeling really stressed about my exams...",
            lines=3
        )

    generate_btn = gr.Button("Generate Music 🎶", variant="primary")

    mood_output = gr.Markdown(label="Detected Mood")
    audio_output = gr.Audio(label="Your Music", type="filepath")

    generate_btn.click(
        fn=run_pipeline,
        inputs=[text_input],
        outputs=[mood_output, audio_output]
    )

    gr.Examples(
        examples=[
            ["I'm so happy today, everything is going great!"],
            ["I feel lonely and a bit down lately."],
            ["I'm absolutely furious at how things turned out."],
            ["Just chilling, nothing much going on."],
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch()
