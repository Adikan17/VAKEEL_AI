# preprocess/text_cleaner.py
import re, hashlib
import spacy
nlp = spacy.load("en_core_web_sm")  # or a legal-tuned model if you have

def clean_text(s):
    s = re.sub(r"\s+", " ", s).strip()
    # remove boilerplate disclaimers or nav patterns (site-specific)
    return s

def extract_entities(text):
    doc = nlp(text)
    ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
    return ents

def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
