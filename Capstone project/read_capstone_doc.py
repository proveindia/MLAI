from docx import Document
import os

def read_docx(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        doc = Document(file_path)
        print(f"--- Content of {os.path.basename(file_path)} ---")
        for para in doc.paragraphs:
            if para.text.strip():
                print(para.text)
        print("--- End of Content ---")
    except Exception as e:
        print(f"Error reading docx: {e}")

if __name__ == "__main__":
    file_path = r"c:\Study\Berkeley_modules\Capstone project\Amazon Sale Prediction.docx"
    read_docx(file_path)
