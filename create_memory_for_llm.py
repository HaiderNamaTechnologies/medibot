from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Define the data path
DATA_PATH = "data/"

# Step 1: Load raw PDFs
def load_pdf_files(data):
    # Use the correct glob pattern (no trailing space)
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data) 
    return text_chunks
text_chunks = create_chunks(extracted_data= documents)
#print("Lenght of text chunks: ", len(text_chunks))

#Step 3: Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DP_FAISS_PATH = "vectorestore/dp_faiss"
dp = FAISS.from_documents(text_chunks, embedding_model)
dp.save_local(DP_FAISS_PATH)    