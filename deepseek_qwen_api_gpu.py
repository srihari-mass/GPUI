from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "deepseek/qwen"  # Change this to the actual model path if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Define the input structure for requests
class InputText(BaseModel):
    text: str

# Endpoint for inference
@app.post("/predict")
async def predict(input_data: InputText):
    # Tokenize the input text
    inputs = tokenizer(input_data.text, return_tensors="pt").to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Assuming the output is logits for classification (adjust accordingly)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

    # Return the result
    return {"predicted_class": predicted_class, "logits": logits.tolist()}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
