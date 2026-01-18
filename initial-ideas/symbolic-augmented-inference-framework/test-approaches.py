# core
print("[DEBUG] Starting script execution...")
print("[DEBUG] Importing core libraries...")
import os
import random
import json
import statistics
from typing import List, Dict
print("[DEBUG] Core libraries imported successfully")

# environment variables
print("[DEBUG] Importing dotenv...")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True, verbose=True)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
print("[DEBUG] Environment variables loaded")

# huggingface hub
print("[DEBUG] Importing huggingface_hub...")
from huggingface_hub import login
print("[DEBUG] Logging in to Hugging Face...")
login(token=HF_TOKEN)
print("[DEBUG] Hugging Face login successful")
# LM
print("[DEBUG] Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
print("[DEBUG] Transformers imported successfully")

# logging
print("[DEBUG] Importing mlflow...")
import mlflow
print("[DEBUG] All imports completed")

MODEL_NAME = "google/gemma-3-1b-it"  # 1.1B parameter model
print(f"[DEBUG] Model name set to: {MODEL_NAME}")

print("[DEBUG] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("[DEBUG] Tokenizer loaded successfully")

print("[DEBUG] Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("[DEBUG] Model loaded successfully")

print("[DEBUG] Creating text-generation pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1
)
print("[DEBUG] Pipeline created successfully")

print("[DEBUG] Initializing knowledge graph...")
KG = {
    "Alice": {
        "works_at": "CompanyX"
    },
    "CompanyX": {
        "located_in": "Berlin"
    }
}
print(f"[DEBUG] Knowledge graph initialized: {KG}")

GROUND_TRUTH_ANSWER = "Alice works at CompanyX, which is located in Berlin."
print(f"[DEBUG] Ground truth answer: {GROUND_TRUTH_ANSWER}")

def rag_prompt():
    print("[DEBUG] Entering rag_prompt()")
    context = (
        "Alice is an employee. CompanyX is a well known company. "
        "Berlin is a city in Europe."
    )
    print(f"[DEBUG] RAG context created: {context}")

    prompt = f"""
Context:
{context}

Question:
Where does Alice work and where is that company located?

Answer:
"""
    print(f"[DEBUG] RAG prompt created (length: {len(prompt)} chars)")
    print("[DEBUG] Exiting rag_prompt()")
    return prompt

def kg_traversal_prompt():
    print("[DEBUG] Entering kg_traversal_prompt()")
    prompt = f"""
You are given the following relations:
Alice works_at CompanyX
CompanyX located_in Berlin

Question:
Traverse the relations step by step and answer:
Where does Alice work and where is that company located?

Answer:
"""
    print(f"[DEBUG] KG traversal prompt created (length: {len(prompt)} chars)")
    print("[DEBUG] Exiting kg_traversal_prompt()")
    return prompt

def saif_prompt():
    print("[DEBUG] Entering saif_prompt()")
    symbolic_facts = {
        "entity": "Alice",
        "works_at": "CompanyX",
        "company_location": "Berlin"
    }
    print(f"[DEBUG] SAIF symbolic facts: {symbolic_facts}")

    prompt = f"""
Structured Knowledge (do not reason about relations):
{json.dumps(symbolic_facts)}

Question:
Where does Alice work and where is that company located?

Answer:
"""
    print(f"[DEBUG] SAIF prompt created (length: {len(prompt)} chars)")
    print("[DEBUG] Exiting saif_prompt()")
    return prompt

def generate(prompt, runs=10):
    print(f"[DEBUG] Entering generate() with runs={runs}")
    print(f"[DEBUG] Prompt preview: {prompt[:100]}...")
    outputs = []
    for i in range(runs):
        print(f"[DEBUG] Generating output {i+1}/{runs}...")
        out = generator(
            prompt,
            max_length=120,
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )[0]["generated_text"]
        print(f"[DEBUG] Generated output {i+1} (length: {len(out)} chars)")
        outputs.append(out)
    print(f"[DEBUG] Exiting generate() with {len(outputs)} outputs")
    return outputs


def evaluate(outputs: List[str]) -> Dict:
    print(f"[DEBUG] Entering evaluate() with {len(outputs)} outputs")
    print("[DEBUG] Calculating exact matches...")
    exact_matches = sum(
        GROUND_TRUTH_ANSWER.lower() in o.lower() for o in outputs
    )
    print(f"[DEBUG] Exact matches found: {exact_matches}/{len(outputs)}")

    print("[DEBUG] Calculating hallucinations...")
    hallucinations = sum(
        any(x in o.lower() for x in ["paris", "google", "london"])
        for o in outputs
    )
    print(f"[DEBUG] Hallucinations found: {hallucinations}/{len(outputs)}")

    metrics = {
        "accuracy": exact_matches / len(outputs),
        "hallucination_rate": hallucinations / len(outputs),
        "raw_outputs": outputs
    }
    print(f"[DEBUG] Evaluation metrics: accuracy={metrics['accuracy']}, hallucination_rate={metrics['hallucination_rate']}")
    print("[DEBUG] Exiting evaluate()")
    return metrics


print("[DEBUG] Setting MLflow experiment...")
mlflow.set_experiment("SAIF_Feasibility_Study")
print("[DEBUG] MLflow experiment set to: SAIF_Feasibility_Study")

print("[DEBUG] Starting MLflow run...")
with mlflow.start_run():
    print("[DEBUG] MLflow run started")

    print("\n[DEBUG] ===== GENERATING RAG OUTPUTS =====")
    rag_outputs = generate(rag_prompt())
    print(f"[DEBUG] RAG outputs generated: {len(rag_outputs)} outputs")
    
    print("\n[DEBUG] ===== GENERATING KG OUTPUTS =====")
    kg_outputs = generate(kg_traversal_prompt())
    print(f"[DEBUG] KG outputs generated: {len(kg_outputs)} outputs")
    
    print("\n[DEBUG] ===== GENERATING SAIF OUTPUTS =====")
    saif_outputs = generate(saif_prompt())
    print(f"[DEBUG] SAIF outputs generated: {len(saif_outputs)} outputs")

    print("\n[DEBUG] ===== EVALUATING RAG OUTPUTS =====")
    rag_metrics = evaluate(rag_outputs)
    print(f"[DEBUG] RAG evaluation complete")
    
    print("\n[DEBUG] ===== EVALUATING KG OUTPUTS =====")
    kg_metrics = evaluate(kg_outputs)
    print(f"[DEBUG] KG evaluation complete")
    
    print("\n[DEBUG] ===== EVALUATING SAIF OUTPUTS =====")
    saif_metrics = evaluate(saif_outputs)
    print(f"[DEBUG] SAIF evaluation complete")
    print(f"[DEBUG] SAIF evaluation complete")

    print("\n[DEBUG] ===== LOGGING METRICS TO MLFLOW =====")
    metrics_to_log = {
        "rag_accuracy": rag_metrics["accuracy"],
        "rag_hallucination": rag_metrics["hallucination_rate"],
        "kg_accuracy": kg_metrics["accuracy"],
        "kg_hallucination": kg_metrics["hallucination_rate"],
        "saif_accuracy": saif_metrics["accuracy"],
        "saif_hallucination": saif_metrics["hallucination_rate"],
    }
    print(f"[DEBUG] Metrics to log: {metrics_to_log}")
    mlflow.log_metrics(metrics_to_log)
    print("[DEBUG] Metrics logged successfully")

    print("[DEBUG] Logging RAG outputs to MLflow...")
    mlflow.log_text("\n\n".join(rag_outputs), "rag_outputs.txt")
    print("[DEBUG] RAG outputs logged")
    
    print("[DEBUG] Logging KG outputs to MLflow...")
    mlflow.log_text("\n\n".join(kg_outputs), "kg_outputs.txt")
    print("[DEBUG] KG outputs logged")
    
    print("[DEBUG] Logging SAIF outputs to MLflow...")
    mlflow.log_text("\n\n".join(saif_outputs), "saif_outputs.txt")
    print("[DEBUG] SAIF outputs logged")
    
    print("\n[DEBUG] ===== ALL RESULTS LOGGED TO MLFLOW =====")
    print("Logged results to MLflow.")
    print("[DEBUG] Script execution completed successfully!")