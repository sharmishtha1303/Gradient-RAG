import pdfplumber
import os
import re

# Input and output paths
input_pdf = "raw_data/ai_ml/AI_notes.pdf"
output_txt = "data_clean/ai_ml_clean.txt"

# Create output folder if not exists
os.makedirs("data_clean", exist_ok=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\b\d{2,6}\b', '', text)     # Remove long numbers (word indexes)
    text = re.sub(r'Page \d+', '', text)        # Remove page numbers
    text = re.sub(r'[\n\s]+', ' ', text)        # Remove extra newlines/spaces
    text = re.sub(r'[^a-zA-Z0-9.,;:?!()\[\] ]', '', text)  # Remove junk symbols
    return text.strip()

text = ""
with pdfplumber.open(input_pdf) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

# Clean text
text = clean_text(text)

# Save cleaned text
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ PDF converted and cleaned successfully!")
