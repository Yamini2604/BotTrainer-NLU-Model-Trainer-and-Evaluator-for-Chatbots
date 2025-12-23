# backend/services/entity_suggester.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
from utils.mongo import db

# Load HF NER model once
HF_MODEL = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForTokenClassification.from_pretrained(HF_MODEL)
hf_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# load spaCy fallback
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def dedupe(entities):
    seen = set()
    out = []
    for e in entities:
        key = (e["start"], e["end"], e["label"], e["value"].lower())
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

def suggest_entities(text: str, workspace_id: str):
    results = []

    # 1) HF NER
    try:
        hf = hf_ner(text)
        for ent in hf:
            results.append({
                "start": ent["start"],
                "end": ent["end"],
                "label": ent["entity_group"],
                "value": text[ent["start"]:ent["end"]]
            })
    except Exception:
        pass

    # 2) spaCy NER fallback
    try:
        doc = nlp(text)
        for ent in doc.ents:
            results.append({
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_,
                "value": ent.text
            })
    except Exception:
        pass

    # 3) If nothing found, try to find simple tokens from dataset items (if dataset contains values)
    if not results:
        ds = db.datasets.find_one({"workspace_id": workspace_id}) or {}
        for item in ds.get("items", []):
            # no entities present normally in raw dataset; so skip this step if absent.
            pass

    # dedupe and sort
    results = dedupe(results)
    results = sorted(results, key=lambda x: x["start"])
    return results
