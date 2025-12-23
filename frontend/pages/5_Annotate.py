import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("Annotate")

ws = st.session_state.get("selected_workspace")
if not ws:
    st.warning("Open workspace first")
    st.stop()

workspace_id = ws.get("id")

dataset = st.session_state.get("current_dataset", [])
if not dataset:
    st.warning("No dataset loaded. Upload first")
    st.stop()

idx = st.number_input("Index", min_value=0, max_value=len(dataset)-1, value=0)
item = dataset[idx]
text = item.get("text", "")
st.subheader("Query")
st.write(text)

# ---- Zero-shot Intent Suggestion ----
suggested_intent = ""
try:
    res = requests.get(
        f"{API}/suggest/nlp/suggest-intent",
        params={"workspace_id": workspace_id, "q": text}
    )
    if res.ok:
        resp = res.json().get("intent", {})
        suggested_intent = resp.get("intent", "")
except:
    suggested_intent = ""

intent = st.text_input("Intent", value=suggested_intent)

# ---- Zero-shot Entity Suggestion ----
entity_string = ""
try:
    res = requests.get(
        f"{API}/suggest/nlp/suggest-entities",
        params={"workspace_id": workspace_id, "q": text}
    )
    if res.ok:
        ents = res.json().get("entities", [])
        # Convert entity objects to simple editable text format
        entity_string = ", ".join(
            f"{text[e['start']:e['end']]}:{e['label']}" for e in ents
        )
except:
    entity_string = ""

entities_input = st.text_area(
    "Entities (editable)  \nFormat: value:TYPE, value2:TYPE",
    value=entity_string
)

st.write("---")

# ---- Save Handler ----
if st.button("Save Annotation"):
    final_entities = []

    for part in entities_input.split(","):
        if ":" in part:
            value, label = part.strip().split(":")
            value = value.strip()
            label = label.strip()

            # Find first matching position in text
            start = text.lower().find(value.lower())
            if start != -1:
                end = start + len(value)
                final_entities.append({"start": start, "end": end, "entity": label})

    payload = {
        "workspace_id": workspace_id,
        "text": text,
        "intent": intent,
        "entities": final_entities
    }

    r = requests.post(f"{API}/annotation/save", json=payload)

    if r.ok:
        st.success("Annotation saved!")
    else:
        st.error("Failed to save!")
