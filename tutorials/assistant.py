# Tutorial: https://www.mlq.ai/gpt-3-document-assistant-langchain/

'''Installs required:
brew install libmagic
brew install poppler
brew install tesseract
pip install google-colab # TODO This is not installing
'''
import langchain
import openai
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import magic
import nltk
nltk.download('punkt')

# os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY" # already set in ~/.bash_profile

#loader = UnstructuredFileLoader('/Users/fettermania/Desktop/dev/llm/langchain-doc-walkthrough/tutorials/docs/self-ask.pdf', mode="elements")
#loader = UnstructuredFileLoader('/Users/fettermania/Desktop/dev/llm/langchain-doc-walkthrough/tutorials/docs/gallian2023_ch10.pdf', mode="elements")
loader = UnstructuredFileLoader('/Users/fettermania/Desktop/dev/llm/langchain-doc-walkthrough/tutorials/docs/gallian2023.pdf', mode="elements")
# Note: Somehow this loads like 364 documents?  Looks like a 'document' here is like a paragraph  
documents = loader.load()

print (f'You have {len(documents)} document(s) in your data')

# print(documents[:5])

# Split into chunks of size < 1000
# TODO: This doesn't seem to change with the chunk_size
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = text_splitter.split_documents(documents)

print("*** NOTE OpenAIEmbeddings is a universal object relating text string closeness")
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

print("*** NOTE Generate embeddings for text chunks and store efficiently as a Chroma object")
docsearch = Chroma.from_documents(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

query = "Give a simple homomorphism"

print ("*** NOTE Asking question " + query)
print ("**** NOTE Asked on a model with docsearch built from an input doc and OpenAI Embeddings")
print ("**** ... powering a VectorDBQA object to do question answering")
print(qa({'query' : query}))

print ("*** NOTE Asking question " + query)
print ("**** NOTE Asked on generic LLM model")
untrained_llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
print(untrained_llm(query))

