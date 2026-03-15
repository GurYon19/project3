from docx import Document

def extract_text(doc_path):
    doc = Document(doc_path)
    for i, p in enumerate(doc.paragraphs):
        has_drawing = 'w:drawing' in p._element.xml
        if has_drawing:
            print(f"Para {i} HAS DRAWING. Text: {p.text}")

if __name__ == "__main__":
    extract_text('Part1-2_Final_Report.docx')
