from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize FastAPI
app = FastAPI(title="Medical KG Triple Extraction API")

# Load model and tokenizer
model_name = "Qwen/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Pydantic model for input
class PatientChartInput(BaseModel):
    text: str
    max_tokens: int = 512

# Function to extract triples
def extract_medical_kg_triples(text, max_tokens=512):
    prompt = f"""
You are a medical expert. From the following patient chart, extract all factual triples in the format (Subject, Predicate, Object).

Only provide valid medical information like diagnoses, medications, symptoms, treatments, etc. Do not include any false or speculative information.

Text:
\"\"\"{text}\"\"\"

Output format:
(Subject, Predicate, Object)
Only return triples with factual information.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the response to return only triples
    if "Output format:" in response:
        response = response.split("Output format:")[-1].strip()

    return response

# Function to convert triples to TSV format
def triples_to_tsv(triples_text):
    tsv_output = []
    for line in triples_text.strip().split("\n"):
        if line.startswith("(") and line.endswith(")"):
            try:
                s, p, o = [x.strip() for x in line[1:-1].split(",")]
                tsv_output.append(f"{s}\t{p}\t{o}")
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return "\n".join(tsv_output)

# FastAPI endpoint
@app.post("/extract-triples")
async def extract_kg(input_data: PatientChartInput):
    try:
        # Extract triples from the text
        triples = extract_medical_kg_triples(input_data.text, input_data.max_tokens)
        
        # Convert the triples to TSV format
        tsv_output = triples_to_tsv(triples)

        return {
            "triples": tsv_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
