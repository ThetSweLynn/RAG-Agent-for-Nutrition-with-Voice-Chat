import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import json
import textwrap


# Setting up environment
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Integrating with LangChain
# RAG
# Integrate with Vector Stores
persist_directory = 'db'
vector_store_exists = os.path.exists(persist_directory)

# Set up embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if vector_store_exists:
    # Load the vector store if it already exists
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    # Load data and create vector store if it doesn't exist
    loader = DirectoryLoader('Nutrition Data', glob='./*.pdf', loader_cls=PyPDFLoader)
    raw_data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    nutrition_data = text_splitter.split_documents(raw_data)
    
    vectordb = Chroma.from_documents(documents=nutrition_data, 
                                     embedding_function=embeddings,
                                     persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k":5})  # Retrieve top 5

# Retrieval QA
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7)

# Custom memory handler to save chat history to a file
class FileBasedMemory:
    def __init__(self, memory_file='chat_history.json'):
        self.memory_file = memory_file
        self.history = self.load_memory()

    def load_memory(self):
        """Load conversation history from file, if exists."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as file:
                return json.load(file)
        else:
            return []

    def save_memory(self):
        """Save current conversation history to file."""
        with open(self.memory_file, 'w') as file:
            json.dump(self.history, file)

    def append_to_history(self, user_input, bot_response):
        """Append the user input and bot response to memory."""
        self.history.append({'user': user_input, 'bot': bot_response})
        self.save_memory()

    def get_history(self):
        """Retrieve the entire conversation history."""
        return '\n'.join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in self.history])


# Initialize file-based memory
memory = FileBasedMemory(memory_file='chat_history.json')

# Set up RetrievalQA with memory context
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Function to wrap text to preserve newlines
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    final_response = wrap_text_preserve_newlines(llm_response['result'])
    return final_response

# To call the RAG agent with file-based memory
def call_rag_agent(query):
    # Retrieve the memory history from the file
    conversation_history = memory.get_history()
    modified_query = f"{conversation_history}\n\nUser: {query}"

    # Get response from RAG agent
    response = qa_chain.invoke(modified_query)
    final_response = process_llm_response(response)

    # Save the conversation to the memory file
    memory.append_to_history(query, final_response)
    
    return final_response

