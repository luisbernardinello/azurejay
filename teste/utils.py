from io import BytesIO
import soundfile
import numpy as np
import tiktoken
from loguru import logger
from schemas import SupportedTextModels, price_table

def audio_array_to_buffer(audio_array: np.ndarray, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav")
    buffer.seek(0)
    return buffer

def count_tokens(text: str | None) -> int:
    if text is None:
        logger.warning("Response is None. Assuming 0 tokens used")
        return 0
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

def calculate_usage_costs(
    prompt: str,
    response: str | None,
    model: SupportedTextModels,
) -> tuple[float, float, float]:
    if model not in price_table:
        # raise at runtime - in case someone ignores type errors
        raise ValueError(f"Cost calculation is not supported for {model} model.")
    price = price_table[model]
    req_costs = price * count_tokens(prompt) / 1000
    res_costs = price * count_tokens(response) / 1000
    total_costs = req_costs + res_costs
    return req_costs, res_costs, total_costs