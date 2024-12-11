import argparse
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline, TextStreamer, BitsAndBytesConfig # !pip install -q requests torch bitsandbytes transformers sentencepiece accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain_community.vectorstores import FAISS
import gradio as gr
import glob
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# 
hf_token =os.getenv('HUGGINGFACE_TOKEN')
model_name = os.getenv('MODEL_NAME')
# 
login(hf_token, add_to_git_credential=True)
# 
text_loader_kwargs = {'encoding': 'utf-8'}

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

def read_docs(path):
    folders = glob.glob(path)
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
    return documents

# 
def process_llms(folder_path):

    # Load a language model for generation
    llm_name = model_name
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_name)

    # 
    pipe = pipeline(
        "text-generation", model=llm_model, tokenizer=llm_tokenizer, max_new_tokens=10,device='mps'
    )
    # Wrap the LLM with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)


    # Load an embedding model for retrieval
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Use RecursiveCharacterTextSplitter instead of CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(read_docs(folder_path))
    vector_store = FAISS.from_documents(chunks, embedding=embedding_model)
    # 

    # set up the conversation memory for the chat
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    
    return conversation_chain

def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(
        description='LLMs with HuggingFace',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--folders_path',
        help='Path to the input file'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the arguments in your program
    
    conversation_chain = process_llms(args.folders_path)
    def chat(question, history):
        result = conversation_chain.invoke({"question": question})
        return result['answer'].split('Helpful Answer:')[-1].strip()

    view = gr.ChatInterface(chat).launch()

if __name__ == '__main__':
    main()

