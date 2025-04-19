from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# Initialize FastAPI app
app = FastAPI()

# Model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model with accelerate for multi-GPU
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device_map = infer_auto_device_map(
    model,
    max_memory={i: "24GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["QWenBlock"],  # Adjust based on model architecture
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # or bfloat16 depending on GPU
)
# Define request input structure
class InputText(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(input_data: InputText):
    # Tokenize input
    inputs = tokenizer(input_data.prompt, return_tensors="pt").to(model.device)

    # Generate text
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=input_data.max_tokens,
            temperature=input_data.temperature,
            do_sample=True,
            top_p=0.95
        )

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
