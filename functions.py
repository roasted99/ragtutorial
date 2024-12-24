import os
import ollama
from pypdf import PdfReader 
import re

def read_pdf_file(path):
  text_content = {}
  directory = os.path.join(path)
  
  for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
      file_path = os.path.join(directory, filename)
      
      with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        text = ""
        
        for page in reader.pages:
          text += page.extract_text()
          
        text_content[filename] = text
        
  return text_content

def chunk_splitter(text, chunk_size=100):
  words = re.findall(r'\S+', text)
  chunks = []
  current_chunk = []
  word_count = 0
  
  for word in words:
    current_chunk.append(word)
    word_count += 1
    
    if word_count >= chunk_size:
      chunks.append(' '.join(current_chunk))
      current_chunk = []
      word_count = 0
      
    if current_chunk:
      chunks.append(' '.join(current_chunk))
      
  return chunks

def get_embedding(chunks):
  embeds = ollama.embed(model="nomic-embed-text", input=chunks)
  return embeds.get('embeddings', [])
  