from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import random
import spacy
from spacy.training.example import Example
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from .annotation import get_annotations_by_workspace
from utils.mongo import db
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import datetime

router = APIRouter()

trained_models: Dict[str, Dict[str, Any]] = {}

class Annotation(BaseModel):
    text: str
    intent: str
    entities: List[Dict[str, Any]]


# -----------------------------
# Helper to update a dataset item field (safe replace approach)
# -----------------------------
def _update_dataset_item_field(workspace_id: str, text: str, field_name: str, field_value: Any):
    """
    Find first item in db.datasets[workspace_id].items where item['text'] == text
    and the field_name does not already exist; set field_name to field_value.
    Uses read-modify-write replace_one to be robust across Mongo versions.
    """
    ds = db.datasets.find_one({"workspace_id": workspace_id})
    if not ds or "items" not in ds:
        return False

    items = ds["items"]
    for idx, it in enumerate(items):
        if isinstance(it.get("text"), str) and it.get("text").strip() == text.strip():
            # only set if not present to avoid overwriting previous results
            if it.get(field_name) is None:
                items[idx][field_name] = field_value
                db.datasets.replace_one({"workspace_id": workspace_id}, {"workspace_id": workspace_id, "items": items})
                return True
            else:
                # already has the field, skip
                return False
    return False


# -----------------------------
# spaCy TRAIN
# -----------------------------
@router.post("/train")
def train_model(
    workspace_id: str = Query(...),
    train_split_ratio: float = Query(0.8),
    debug: bool = False
):
    annotations = get_annotations_by_workspace(workspace_id)
    if not annotations:
        raise HTTPException(status_code=404, detail="No annotations found.")

    all_intents = list({a["intent"] for a in annotations})
    data = []
    for ann in annotations:
        text = ann["text"]
        intent = ann["intent"]
        cats = {lbl: 0.0 for lbl in all_intents}
        cats[intent] = 1.0
        entities = [(e["start"], e["end"], e["label"]) for e in ann.get("entities", [])]
        data.append((text, {"entities": entities, "cats": cats}))

    random.shuffle(data)
    split = int(len(data) * train_split_ratio)
    train_data = data[:split]
    test_data = data[split:]

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    textcat = nlp.add_pipe("textcat")

    for text, ann in train_data:
        for start, end, label in ann["entities"]:
            ner.add_label(label)
    for intent_label in all_intents:
        textcat.add_label(intent_label)

    other_pipes = [p for p in nlp.pipe_names if p not in ("ner", "textcat")]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for it in range(20):
            random.shuffle(train_data)
            losses = {}
            batches = spacy.util.minibatch(train_data, size=4)
            for batch in batches:
                texts, anns = zip(*batch)
                examples = [Example.from_dict(nlp.make_doc(t), a) for t, a in zip(texts, anns)]
                nlp.update(examples, drop=0.1, sgd=optimizer, losses=losses)
            if debug:
                print(f"Iteration {it+1} → Losses = {losses}")

    trained_models.setdefault(workspace_id, {})["spacy"] = {
        "nlp": nlp,
        "test_data": test_data,
        "all_intents": all_intents
    }

    # === Persist train_result into dataset items for train split ===
    train_stamp = {
        "model": "spacy",
        "used_in_split": "train",
        "train_samples": len(train_data),
        "created_at": datetime.datetime.utcnow()
    }
    for text, _ in train_data:
        _update_dataset_item_field(workspace_id, text, "train_result", train_stamp)

    # record train history in DB
    train_doc = {
        "workspace_id": workspace_id,
        "model": "spacy",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "metrics": {},
        "created_at": datetime.datetime.utcnow()
    }
    db.train_history.insert_one(train_doc)

    return {
        "message": "spaCy training completed",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "intents": all_intents
    }


# -----------------------------
# spaCy TEST
# -----------------------------
@router.post("/test")
def test_model(workspace_id: str = Query(...), debug: bool = False):
    if workspace_id not in trained_models or "spacy" not in trained_models[workspace_id]:
        raise HTTPException(404, "spaCy model not trained yet for this workspace.")

    info = trained_models[workspace_id]["spacy"]
    nlp = info["nlp"]
    test_data = info["test_data"]
    all_intents = info["all_intents"]

    if not test_data:
        raise HTTPException(400, "No test data found for spaCy.")

    y_true = []
    y_pred = []
    detailed_results = []

    for text, ann in test_data:
        true_intent = max(ann["cats"], key=lambda k: ann["cats"][k])
        doc = nlp(text)
        pred_intent = max(doc.cats, key=doc.cats.get) if doc.cats else "unknown"
        confidence = float(doc.cats.get(pred_intent, 0.0)) if doc.cats else 0.0

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        detailed_results.append({
            "text": text,
            "true_intent": true_intent,
            "predicted_intent": pred_intent,
            "confidence": confidence
        })

        if debug:
            print(f"TEXT: {text}\nTRUE: {true_intent}\nPRED: {doc.cats}\n")

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_intents, average='weighted', zero_division=0
    )
    cls_report = classification_report(y_true, y_pred, labels=all_intents, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=all_intents).tolist()

    # === Persist per-item test_result inside dataset items ===
    for res in detailed_results:
        text = res["text"]
        test_result = {
            "model": "spacy",
            "predicted": res["predicted_intent"],
            "confidence": res["confidence"],
            "correct": res["predicted_intent"] == res["true_intent"],
            "tested_at": datetime.datetime.utcnow()
        }
        _update_dataset_item_field(workspace_id, text, "test_result", test_result)

    # record test run in DB
    test_doc = {
        "workspace_id": workspace_id,
        "model": "spacy",
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": cm,
        "all_intents": all_intents,
        "samples_tested": len(test_data),
        "detailed_results": detailed_results,
        "created_at": datetime.datetime.utcnow()
    }
    db.test_history.insert_one(test_doc)

    return {
        "model": "spacy",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "all_intents": all_intents,
        "samples_tested": len(test_data),
        "detailed_results": detailed_results
    }


# -----------------------------
# RASA-LIKE TRAIN
# -----------------------------
@router.post("/train_rasa")
def train_rasa_like(workspace_id: str = Query(...), train_split_ratio: float = Query(0.8), debug: bool = False):
    annotations = get_annotations_by_workspace(workspace_id)
    if not annotations:
        raise HTTPException(status_code=404, detail="No annotations found.")

    all_intents = list({a["intent"] for a in annotations})
    X = []
    y = []
    data_pairs = []
    for ann in annotations:
        text = ann["text"]
        intent = ann["intent"]
        X.append(text)
        y.append(intent)
        cats = {lbl: 1.0 if lbl == intent else 0.0 for lbl in all_intents}
        entities = [(e["start"], e["end"], e["label"]) for e in ann.get("entities", [])]
        data_pairs.append((text, {"entities": entities, "cats": cats}))

    combined = list(zip(X, y, data_pairs))
    random.shuffle(combined)
    split = int(len(combined) * train_split_ratio)
    train_comb = combined[:split]
    test_comb = combined[split:]

    if len(train_comb) == 0:
        raise HTTPException(400, "Not enough samples to train Rasa-like model.")

    X_train = [t for t, lab, _ in train_comb]
    y_train = [lab for t, lab, _ in train_comb]
    X_test_pairs = [pair for _, _, pair in test_comb]

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_t = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_t, y_train)

    trained_models.setdefault(workspace_id, {})["rasa"] = {
        "model": clf,
        "vectorizer": vectorizer,
        "test_data": X_test_pairs,
        "all_intents": all_intents
    }

    # === Persist train_result into dataset items for Rasa train split ===
    train_stamp = {
        "model": "rasa_like",
        "used_in_split": "train",
        "train_samples": len(X_train),
        "created_at": datetime.datetime.utcnow()
    }
    for t in X_train:
        _update_dataset_item_field(workspace_id, t, "train_result", train_stamp)

    # record train history
    train_doc = {
        "workspace_id": workspace_id,
        "model": "rasa_like",
        "train_samples": len(X_train),
        "test_samples": len(X_test_pairs),
        "metrics": {},
        "created_at": datetime.datetime.utcnow()
    }
    db.train_history.insert_one(train_doc)

    return {
        "message": "Rasa-like training completed",
        "train_samples": len(X_train),
        "test_samples": len(X_test_pairs),
        "intents": all_intents
    }


# -----------------------------
# RASA-LIKE TEST
# -----------------------------
@router.post("/test_rasa")
def test_rasa_like(workspace_id: str = Query(...), debug: bool = False):
    if workspace_id not in trained_models or "rasa" not in trained_models[workspace_id]:
        raise HTTPException(404, "Rasa-like model not trained yet for this workspace.")

    info = trained_models[workspace_id]["rasa"]
    clf = info["model"]
    vectorizer = info["vectorizer"]
    test_data = info["test_data"]
    all_intents = info["all_intents"]

    if not test_data:
        raise HTTPException(400, "No test data found for Rasa-like model.")

    y_true = []
    y_pred = []
    detailed_results = []

    for text, ann in test_data:
        true_intent = max(ann["cats"], key=lambda k: ann["cats"][k])
        X_vec = vectorizer.transform([text])
        pred = clf.predict(X_vec)[0]
        probs = clf.predict_proba(X_vec)[0] if hasattr(clf, "predict_proba") else None
        confidence = float(max(probs)) if probs is not None else 0.0

        y_true.append(true_intent)
        y_pred.append(pred)

        detailed_results.append({
            "text": text,
            "true_intent": true_intent,
            "predicted_intent": pred,
            "confidence": confidence
        })

        if debug:
            print(f"RASA-LIKE TEST | TEXT: {text}\nTRUE: {true_intent}\nPRED: {pred}")

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_intents, average='weighted', zero_division=0
    )
    cls_report = classification_report(y_true, y_pred, labels=all_intents, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=all_intents).tolist()

    # === Persist per-item test_result inside dataset items ===
    for res in detailed_results:
        text = res["text"]
        test_result = {
            "model": "rasa_like",
            "predicted": res["predicted_intent"],
            "confidence": res["confidence"],
            "correct": res["predicted_intent"] == res["true_intent"],
            "tested_at": datetime.datetime.utcnow()
        }
        _update_dataset_item_field(workspace_id, text, "test_result", test_result)

    # record test history
    test_doc = {
        "workspace_id": workspace_id,
        "model": "rasa_like",
        "metrics": {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1},
        "confusion_matrix": cm,
        "all_intents": all_intents,
        "samples_tested": len(test_data),
        "detailed_results": detailed_results,
        "created_at": datetime.datetime.utcnow()
    }
    db.test_history.insert_one(test_doc)

    return {
        "model": "rasa_like",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "all_intents": all_intents,
        "samples_tested": len(test_data),
        "detailed_results": detailed_results
    }
