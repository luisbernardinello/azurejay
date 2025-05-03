from datetime import datetime
from typing import Annotated, Literal, TypeAlias
from uuid import uuid4
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    IPvAnyAddress,
)
from utils import count_tokens

VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"]
SupportedTextModels: TypeAlias = Literal["gpt-3.5", "gpt-4"]
TokenCount = Annotated[int, Field(ge=0)]

PriceTable: TypeAlias = dict[SupportedTextModels, float]
price_table: PriceTable = {"gpt-3.5": 0.0030, "gpt-4": 0.0200}

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)]
    ip: Annotated[str, IPvAnyAddress] | None
    content: Annotated[str | None, Field(min_length=0, max_length=10000)]
    created_at: datetime = datetime.now()

class TextModelRequest(BaseModel):
    model: Literal["gpt-3.5-turbo", "gpt-4o"]
    prompt: str
    temperature: float = 0.0

class TextModelResponse(ModelResponse):
    model: SupportedTextModels
    temperature: Annotated[float, Field(ge=0.0, le=1.0, default=0.0)]
    price: Annotated[float, Field(ge=0, default=0.0)]
    
    @property
    @computed_field
    def tokens(self) -> TokenCount:
        return count_tokens(self.content)
    
    @property
    @computed_field
    def cost(self) -> float:
        return self.price * self.tokens