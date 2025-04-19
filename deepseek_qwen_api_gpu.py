from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# Initialize FastAPI app
app = FastAPI()

# Model path
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Load tokenizer and model with multi-GPU support
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # automatically split across GPUs
    torch_dtype=torch.float16
)

# Set up generation pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Define input structure
class InputText(BaseModel):
    text: str
    max_new_tokens: int = 100

# Text generation endpoint
@app.post("/generate")
async def generate(input_data: InputText):
    output = pipeline(
        input_data.text,
        max_new_tokens=input_data.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return {"generated_text": output[0]["generated_text"]}
