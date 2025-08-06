import pdfplumber
import os
import re

input_pdf = "raw_data/ai_ml/AI_notes.pdf"
output_txt = "data_clean/AI_notes.txt"

os.makedirs("data_clean", exist_ok=True)

def is_toc_line(line):
    """Detect Table of Contents or index lines."""
    # If line contains many dots or ends with a number (page number)
    if line.count('.') > 4:
        return True
    if re.match(r'^\s*\d+(\.\d+)+\s+[A-Za-z].*\d+\s*$', line):
        return True
    return False

def clean_text(text):
    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Skip TOC-like noise
        if is_toc_line(line):
            continue
        cleaned_lines.append(line)

    text = ' '.join(cleaned_lines)

    # Basic cleanup
    text = re.sub(r'\(cid:\d+\)', ' ', text)
    text = re.sub(r'Page\s?\d+', '', text)
    text = re.sub(r'Fig\.?\s?\d+', '', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'\. ', '.\n', text)

    return text.strip()

text = ""
with pdfplumber.open(input_pdf) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

text = clean_text(text)

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ PDF converted and cleaned successfully!")