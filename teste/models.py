import torch
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModel,
    BarkProcessor,
    BarkModel,
    Pipeline,
    pipeline,
)
from schemas import VoicePresets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt = "How to set up a FastAPI project?"
system_prompt = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

def load_text_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    return pipe

def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    predictions = pipe(
        prompt,
        temperature=temperature,
        max_new_tokens=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    output = predictions[0]["generated_text"].split("</s>\n<|assistant|>\n")[-1]
    return output

def load_audio_model() -> tuple[BarkProcessor, BarkModel]:
    processor = AutoProcessor.from_pretrained("suno/bark-small", device=device)
    model = AutoModel.from_pretrained("suno/bark-small", device=device)
    return processor, model

def generate_audio(
    processor: BarkProcessor,
    model: BarkModel,
    prompt: str,
    preset: VoicePresets,
) -> tuple[np.array, int]:
    inputs = processor(text=[prompt], return_tensors="pt", voice_preset=preset)
    output = model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    return output, sample_rate