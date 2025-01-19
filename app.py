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



