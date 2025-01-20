from rich import print
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

local_llm = ChatOllama(model="mistral")

def rag(chunks, collection_name):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=embeddings.ollama.OllamaEmbeddings(model="nomic-embed-text"),
    )
    retriever = vectorstore.as_retriever()

    prompt_template = """
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        |prompt
        |local_llm
        |StrOutputParser()
    )

    result = chain.invoke("What is the use of Text Splitting?")
    print(result)

# 1. Character Text Splitting

print("Character Text Splitting")

text = "Character Text Splitting is the process of splitting a text into individual characters. It is useful for various Natural Language Processing tasks such as text classification, sentiment analysis, and named entity recognition."

# Manual Splitting
chunks=[]
chunk_size = 35 # 35 characters per chunk
for i in range(0, len(text), chunk_size):
    chunk=text[i:i+chunk_size]
    chunks.append(chunk)

documents= [Document(page_content=chunk,metadata={"Source":"local"}) for chunk in chunks]
print(documents)



# Automatic Text Splitting
"""
1. The following code demonstrates how to use the langchain text splitter to automatically split the text into chunks.
2. It devides the text into chunks of 35 characters each with an overlap of 5 characters between the chunks.
3. It is fixed size so words might get split between chunks.
"""

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=35, chunk_overlap=5, separator="", strip_whitespace=False)
documents=text_splitter.create_documents([text],metadatas=[{"Source":"local"}])
print(documents)


#2. Recursive Character Text Splitting

print("Recursive Character Text Splitting")

with open("sample_data.txt", "r") as file:
    file_text = file.read()

print(file_text)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
documents=text_splitter.create_documents([file_text],metadatas=[{"Source":"local"}])
print(documents)

#3. Document Specific Splitting - Markdown 

print("Document Specific Splitting - Markdown")

from langchain.text_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size = 40, chunk_overlap=0)
markdown_text = """
# Fun in California

## Driving

Try driving on the 1 down to San Diego

### Food

Make sure to eat a burrito while you're there

## Hiking

Go to Yosemite
"""
print(splitter.create_documents([markdown_text]))

# Document Specific Splitting - Python
from langchain.text_splitter import PythonCodeTextSplitter
python_text = """
def add(a, b):
    return a + b

def subtract(a, b):

    return a - b
"""
splitter = PythonCodeTextSplitter(chunk_size = 40, chunk_overlap=0)
print(splitter.create_documents([python_text]))



#Document Specific Splitting - JavaSript
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

javascript_text = """
function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

let x=myFunction(4, 3);
"""
splitter = RecursiveCharacterTextSplitter.from_language(chunk_size = 60, chunk_overlap=0, language=Language.JS)

print(splitter.create_documents([javascript_text]))


# 4. Semantic Chunking



