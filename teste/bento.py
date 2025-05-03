import bentoml
from models import load_text_model

@bentoml.service(
    resources={"cpu": "4"}, traffic={"timeout": 120}, http={"port": 5000}
)
class Generate:
    def __init__(self) -> None:
        self.text_model = load_text_model()
    
    @bentoml.api(route="/generate/text")
    def generate_text(self, prompt: str) -> str:
        from models import generate_text
        output = generate_text(self.text_model, prompt)
        return output