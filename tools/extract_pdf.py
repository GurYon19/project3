import fitz  # PyMuPDF

def extract_pdf_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text())

extract_pdf_text('Project3_guidelines-3(2).pdf', 'guidelines_full.txt')
