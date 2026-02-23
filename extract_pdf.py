import sys
try:
    import PyPDF2
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

try:
    reader = PyPDF2.PdfReader("Project3_guidelines-3(2).pdf")
    text = '\n'.join(page.extract_text() for page in reader.pages)
    with open("full_guidelines.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully wrote full_guidelines.txt")
except Exception as e:
    print(f"Error: {e}")
