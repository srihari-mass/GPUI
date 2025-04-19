from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Initialize FastAPI app
app = FastAPI()

# Model path
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # automatically split across GPUs
    torch_dtype=torch.float16
)

# Set up generation pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define input structure
class Message(BaseModel):
    role: str
    content: str

class InputMessages(BaseModel):
    messages: list[Message]

# Text generation endpoint
@app.post("/generate")
async def generate(input_data: InputMessages):
    # Construct prompt from messages
    prompt = ""
    for msg in input_data.messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    
    # Generate response
    output = pipe(prompt, max_length=200, do_sample=True, temperature=0.7, top_p=0.9)
    
    # Return generated text
    return {"generated_text": output[0]['generated_text']}
