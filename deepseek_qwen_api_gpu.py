from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import rdflib

# FastAPI app initialization
app = FastAPI()

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically use multiple GPUs if available
    torch_dtype=torch.float16,  # Use FP16 for faster inference
    trust_remote_code=True
)

model.eval()  # Ensure evaluation mode for faster processing

# Input model for patient chart text
class PatientChartInput(BaseModel):
    text: str
    max_tokens: int = 100  # Limit the output to speed up

# Function to extract medical triples
async def extract_medical_kg_triples(text, max_tokens=100):
    prompt = f"""
    You are a medical expert. Extract factual triples in the format (Subject, Predicate, Object) from the following patient chart:
    Text: "{text}"
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the output
    return response.strip()

# Function to build RDF graph
def build_medical_knowledge_graph(triples_text):
    graph = rdflib.Graph()
    triples = []
    for line in triples_text.strip().split("\n"):
        if line.startswith("(") and line.endswith(")"):
            try:
                s, p, o = [x.strip() for x in line[1:-1].split(",")]
                graph.add((
                    rdflib.URIRef(f"http://example.org/{s.replace(' ', '_')}"),
                    rdflib.URIRef(f"http://example.org/{p.replace(' ', '_')}"),
                    rdflib.URIRef(f"http://example.org/{o.replace(' ', '_')}"),
                ))
                triples.append((s, p, o))  # Collect triples for TSV output
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return graph, triples

# Function to convert triples to TSV format
def triples_to_tsv(triples):
    tsv_output = "Subject\tPredicate\tObject\n"
    for triple in triples:
        tsv_output += f"{triple[0]}\t{triple[1]}\t{triple[2]}\n"
    return tsv_output

# FastAPI endpoint for extraction
@app.post("/extract-triples")
async def extract_kg(input_data: PatientChartInput, background_tasks: BackgroundTasks):
    try:
        # Start processing in the background
        background_tasks.add_task(process_kg, input_data)
        return {"message": "Request received. Processing in the background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to process the input in the background and generate triples + TSV
async def process_kg(input_data: PatientChartInput):
    triples = await extract_medical_kg_triples(input_data.text, input_data.max_tokens)
    kg, triples_list = build_medical_knowledge_graph(triples)

    # Convert triples to TSV format
    tsv_output = triples_to_tsv(triples_list)

    # You can either print or save the RDF graph and TSV to a file here if needed
    print("Generated Triples in TSV Format:\n", tsv_output)
    
    return {
        "generated_response": triples,
        "tsv_output": tsv_output
    }
