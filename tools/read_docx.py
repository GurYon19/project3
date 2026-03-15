from docx import Document

def extract_text(doc_path):
    doc = Document(doc_path)
    for i, p in enumerate(doc.paragraphs):
        print(f"Para {i}: {p.text}")

if __name__ == "__main__":
    extract_text('d:\\Reichman\\Computrer Vision Course\\project3\\Part1-2_Final_Report_Formatted.docx')
