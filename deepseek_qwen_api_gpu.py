from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# FastAPI app initialization
app = FastAPI()

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically use multiple GPUs if available
    torch_dtype=torch.float16,  # Use FP16 for faster inference and less memory usage
    trust_remote_code=True
)

model.eval()  # Ensure evaluation mode for faster processing

# Input model for patient chart text
class PatientChartInput(BaseModel):
    text: str
    max_tokens: int = 100  # Limit the output to speed up
    page_length: int = 1000  # Max token length per page (adjust as needed)

# Function to extract medical triples
async def extract_medical_kg_triples(text, max_tokens=100):
    prompt = f"""
    You are a medical expert. Extract factual triples in the format (Subject, Predicate, Object) from the following patient chart:
    Text: "{text}"
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Use batched generation with max tokens per batch for better utilization
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, 
                             do_sample=True,  # Allows sampling (increases diversity)
                             num_beams=5,     # Use beam search to maximize GPU usage
                             temperature=0.7,  # Adjust temperature for better performance
                             top_k=50)        # Limit the top k candidates during sampling

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the output
    return response.strip()

# Function to split document into pages (based on the page_length setting)
def split_document_into_pages(text, page_length=1000):
    pages = []
    while len(text) > page_length:
        pages.append(text[:page_length])
        text = text[page_length:]
    if text:  # Add the remaining part as the last page
        pages.append(text)
    return pages

# Function to convert triples to KGTK format (without using rdflib)
def triples_to_kgtk(triples):
    kgtk_output = "Node1\tLabel\tNode2\n"
    for triple in triples:
        # KGTK format expects the first node as the subject, the predicate as label, and the object as the second node
        kgtk_output += f"{triple[0]}\t{triple[1]}\t{triple[2]}\n"
    return kgtk_output

# Function to process the input in the background and generate triples + KGTK
async def process_kg(input_data: PatientChartInput):
    # Split the document into pages (if necessary)
    pages = split_document_into_pages(input_data.text, input_data.page_length)
    
    all_triples = []
    all_kgtk_output = "Node1\tLabel\tNode2\n"
    
    for page in pages:
        triples = await extract_medical_kg_triples(page, input_data.max_tokens)
        
        # Now we have the triples as a string, let's parse them into a list
        triples_list = []
        for line in triples.strip().split("\n"):
            if line.startswith("(") and line.endswith(")"):
                try:
                    s, p, o = [x.strip() for x in line[1:-1].split(",")]
                    triples_list.append((s, p, o))  # Collect triples for KGTK output
                except Exception as e:
                    print(f"Error processing line: {line}, Error: {e}")
        
        # Convert page triples to KGTK format and append
        kgtk_output = triples_to_kgtk(triples_list)
        all_kgtk_output += kgtk_output
        
        # Append the triples for further use
        all_triples.extend(triples_list)

    # Store the combined triples in memory for querying later
    process_kg.kg_data = [{"Node1": s, "Label": p, "Node2": o} for s, p, o in all_triples]
    
    return {
        "generated_response": all_triples,
        "kgtk_output": all_kgtk_output,
        "kg_data": process_kg.kg_data  # Return the KG data for querying
    }

# FastAPI endpoint for extraction
@app.post("/extract-triples")
async def extract_kg(input_data: PatientChartInput):
    try:
        result = await process_kg(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FastAPI endpoint to query the KG
@app.get("/query-kg")
async def query_kg(subject: str = Query(None), predicate: str = Query(None), object_: str = Query(None)):
    """
    Query the knowledge graph for specific triples.
    - subject: Query by subject
    - predicate: Query by predicate (label)
    - object_: Query by object (node)
    """
    # Example: Let's say we have a global variable that stores the KG data
    # For the sake of this example, assume `process_kg.kg_data` is the knowledge graph generated earlier.
    
    if not hasattr(process_kg, "kg_data"):
        raise HTTPException(status_code=404, detail="No knowledge graph data available. Please process the KG first.")

    # Filter KG data based on the provided query parameters
    result = [
        triple for triple in process_kg.kg_data
        if (subject is None or triple["Node1"] == subject) and
           (predicate is None or triple["Label"] == predicate) and
           (object_ is None or triple["Node2"] == object_)
    ]

    if not result:
        raise HTTPException(status_code=404, detail="No matching triples found.")

    return {"result": result}
