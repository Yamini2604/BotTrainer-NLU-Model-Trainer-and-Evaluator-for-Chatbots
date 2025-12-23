from typing import List, Dict
from utils.mongo import db

def get_annotations_by_workspace(workspace_id: str) -> List[Dict]:
    """
    Fetch annotated data from MongoDB for the given workspace_id.
    Returns a list of dicts with keys: text, intent, entities.
    """
    query = {"workspace_id": workspace_id}
    projection = {"_id": 0, "text": 1, "intent": 1, "entities": 1}
    cursor = db.annotations.find(query, projection)

    results = []
    for doc in cursor:
        if "text" in doc and "intent" in doc and "entities" in doc:
            results.append({
                "text": doc["text"],
                "intent": doc["intent"],
                "entities": doc["entities"],
            })
    return results
