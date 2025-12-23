# backend/services/intent_suggester.py
from transformers import pipeline
from utils.mongo import db

# default candidate intents to use when user has no annotated intents
DEFAULT_INTENTS = [
    "sports_info", "sports_player_info", "sports_team_info",
    "sports_match_schedule", "sports_rankings", "sports_news",
    "sports_statistics", "olympics_info",
    "finance_stock_price", "finance_market_summary",
    "finance_account_info", "finance_interest_rates",
    "finance_money_transfer", "finance_credit_cards",
    "finance_mutual_funds", "finance_expense_analysis",
    "finance_exchange_rates", "finance_bank_security",
    "finance_tax_investments", "finance_crypto_info"
]

# load model once
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_all_intents(workspace_id: str):
    # try previously annotated intents
    annotated_intents = db.annotations.distinct("intent", {"workspace_id": workspace_id})
    annotated_intents = [i for i in annotated_intents if i]
    if not annotated_intents:
        return DEFAULT_INTENTS
    return annotated_intents

def suggest_intent(text: str, workspace_id: str):
    candidate_labels = get_all_intents(workspace_id)
    if not candidate_labels:
        return {"intent": None, "confidence": 0.0}
    res = classifier(sequences=text, candidate_labels=candidate_labels, multi_label=False)
    return {"intent": res["labels"][0], "confidence": float(res["scores"][0])}
