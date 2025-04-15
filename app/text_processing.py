import re
from PyPDF2 import PdfReader
import docx

def extract_text_from_file(filepath):
    """Extract text from various file formats."""
    if filepath.endswith('.pdf'):
        return extract_from_pdf(filepath)
    elif filepath.endswith('.docx'):
        return extract_from_docx(filepath)
    elif filepath.endswith('.txt'):
        return extract_from_txt(filepath)
    else:
        return "Unsupported file format"

def extract_from_pdf(filepath):
    """Extract text from PDF file."""
    text = ""
    with open(filepath, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_from_docx(filepath):
    """Extract text from DOCX file."""
    doc = docx.Document(filepath)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_from_txt(filepath):
    """Extract text from TXT file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    """Clean and preprocess extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()