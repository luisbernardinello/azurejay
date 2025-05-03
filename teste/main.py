from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable
import csv
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, status, Request, Response
from fastapi.responses import StreamingResponse
from openai import OpenAI
from models import (
    load_text_model,
    generate_text,
    load_audio_model,
    generate_audio,
)
from schemas import VoicePresets
from utils import audio_array_to_buffer

openai_client = OpenAI()
system_prompt = "You are a helpful assistant."

models = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Load models during startup
    models["text"] = load_text_model()
    models["audio_processor"], models["audio_model"] = load_audio_model()

    yield

    # Cleanup on shutdown
    models.clear()


app = FastAPI(lifespan=lifespan)

csv_header = [
    "Request ID",
    "Datetime",
    "Endpoint Triggered",
    "Client IP Address",
    "Response Time",
    "Status Code",
    "Successful",
]


@app.middleware("http")
async def monitor_service(
    req: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = uuid4().hex
    request_datetime = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    response: Response = await call_next(req)
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id
    with open("usage.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(csv_header)
        writer.writerow(
            [
                request_id,
                request_datetime,
                req.url,
                req.client.host,
                response_time,
                response.status_code,
                response.status_code < 400,
            ]
        )
    return response


@app.get("/")
def root_controller():
    return {"status": "healthy"}


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me") -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices[0].message.content
    return {"statement": statement}


@app.get("/generate/text")
def serve_language_model_controller(prompt: str) -> str:
    output = generate_text(models["text"], prompt)
    return output


@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_audio_model_controller(
    prompt: str,
    preset: VoicePresets = "v2/en_speaker_1",
):
    output, sample_rate = generate_audio(
        models["audio_processor"], models["audio_model"], prompt, preset
    )
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )


@app.get("/generate/openai/text")
def serve_openai_language_model_controller(prompt: str) -> str | None:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content