import io
from typing import List
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

def process_uploaded_files(uploaded_files) -> List[str]:
    """Process uploaded files and return text chunks"""
    all_text = ""
    
    for uploaded_file in uploaded_files:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                # Process text file
                text = str(uploaded_file.read(), "utf-8")
                all_text += text + "\n\n"
                
            elif file_extension == 'pdf':
                # Process PDF file
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        all_text += text + "\n\n"
                    
            elif file_extension == 'docx':
                # Process Word document
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():  # Only add non-empty paragraphs
                        all_text += paragraph.text + "\n"
                all_text += "\n"
                
            print(f"Successfully processed {uploaded_file.name}")
                
        except Exception as e:
            print(f"Error processing file {uploaded_file.name}: {e}")
            continue
    
    if not all_text.strip():
        print("No text extracted from files")
        return []
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_text(all_text)
    print(f"Created {len(chunks)} text chunks")
    return chunks
