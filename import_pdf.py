import chromadb
from functions import read_pdf_file, chunk_splitter, get_embedding

chroma_client = chromadb.HttpClient(host='localhost', port='8000')
text_pdf_path = "/home/khamlao/Documents/data"
text_data = read_pdf_file(text_pdf_path)

collections = chroma_client.get_or_create_collection(name='first_rag', metadata={'hnsw:space': 'cosine'})
if any(collections.name == 'first_rag' for collection in chroma_client.list_collections()):
  chroma_client.delete_collection('first_rag')
  
  for filename, text in text_data.items():
    chunks = chunk_splitter(text)
    embeds = get_embedding(chunks)
    chunk_number = list(range(len(chunks)))
    ids = [filename + str(index) for index in chunk_number]
    metadatas = [{'source': filename} for index in chunk_number]
    collections.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)