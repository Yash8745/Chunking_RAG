# Text Chunking Techniques: A Comparative Analysis  

This project explores and tests multiple methods of text chunking using various techniques and libraries. Text chunking is a vital step in Natural Language Processing (NLP) for tasks like retrieval-augmented generation (RAG), semantic search, and document processing. The goal is to evaluate how different chunking strategies impact the effectiveness of downstream tasks.  

## Features tested:

- **Character-based Chunking**: Splitting text into fixed-size chunks manually or using `CharacterTextSplitter`.  
- **Recursive Character Chunking**: Handling large documents with recursive chunking logic for optimal text segmentation.  
- **Document-Specific Chunking**: Chunking tailored for formats like Markdown, Python, and JavaScript code.  
- **Semantic Chunking**: Using embeddings for semantically meaningful text segmentation.  
- **Agentic Chunking**: Employing AI-driven approaches for proportion-based and proposition-aware chunking.  


## Installation  
0. Ensure you have Ollama installed:  
   * Go to Ollama's Website and install latest version according to your OS.

1. Clone the repository:  
   ```bash
   git clone https://github.com/Yash8745/Chunking_RAG.git
   ```  
2. Create and activate a virtual environment:  
   ```bash
   conda create -n chunking python=3.11 -y
   conda activate chunking
   ```  
3. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
4. Pull the required models using Ollama:  
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral
   ```  
5. Create a `.env` file for environment-specific configurations.  


## How to Run  

1. Ensure you have the necessary data files, such as `sample_data.txt`, in the project directory.  
2. Run the main script:  
   ```bash
   python main.py
   ```  
3. View the output for each chunking method in the terminal.  

## Directory Structure
```
chunking_rag/
│
├── README.md
├── agentic_chunker.py
├── app.py
├── requirements.txt
└── sample_data.txt
```



## Chunking Methods  

Chunking methods refer to various strategies for breaking down large pieces of text into manageable, meaningful segments. These methods are essential for applications in natural language processing, such as summarization, semantic search, and text generation.


### 1. **Character Text Splitting**  
This method involves dividing text into chunks based on character count. It’s straightforward and commonly used in text preprocessing.  

#### Methods:
- **Manual Splitting**: Text is split into chunks of a fixed size (e.g., 1000 characters) without overlap.  
  **Example**:  
  ```plaintext
  Original Text: "Hi my name is Yash and I am trying to become an machine learning Engineer..."
  Chunk 1: "Hi my name is Yash and I am trying to be"
  Chunk 2: "come an machine learning Engineer..."
  ```

- **Automatic Splitting**: Uses tools like `CharacterTextSplitter` to split text with configurable parameters.  
  **Parameters**:  
  - `chunk_size`: Maximum size of each chunk.  
  - `chunk_overlap`: Number of overlapping characters between consecutive chunks.  

  **Example** (Using `chunk_size=50`, `chunk_overlap=10`):  
  ```plaintext
  Chunk 1: "Hi my name is Yash and I am trying to become an machin"
  Chunk 2: "an machine learning Engineer..."
  ```



### 2. **Recursive Character Text Splitting**  
Recursive splitting is used when the text is too large for fixed-size chunks. It divides text progressively into smaller chunks, ensuring better segmentation.

- **Configurable Parameters**:  
  - `chunk_size`: Size of the final chunks.  
  - `chunk_overlap`: Overlap between chunks to maintain context.  

**Example**:  
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text("Large text input goes here...")
```



### 3. **Document-Specific Chunking**  
This method customizes chunking for different document types to preserve structure and logic.  

#### Types:  

- **Markdown Splitting**: Segments Markdown files while maintaining headings and sections.  
  **Example**:  
  ```markdown
  ### Section 1  
  Content of section 1...

  ### Section 2  
  Content of section 2...
  ```
  Result:  
  - Chunk 1: "### Section 1\nContent of section 1..."  
  - Chunk 2: "### Section 2\nContent of section 2..."  

- **Python Code Splitting**: Splits Python code while respecting logical blocks like functions and classes.  
  **Example**:  
  ```python
  def function1():
      # Function 1 implementation

  def function2():
      # Function 2 implementation
  ```
  Result:  
  - Chunk 1: `def function1():\n    # Function 1 implementation`  
  - Chunk 2: `def function2():\n    # Function 2 implementation`  

- **JavaScript Code Splitting**: Similar to Python, this method respects constructs like functions, objects, and modules.  
  **Example**:  
  ```javascript
  function func1() {
      // Implementation
  }

  function func2() {
      // Implementation
  }
  ```



### 4. **Semantic Chunking**  
Semantic chunking uses embeddings to split text based on meaning rather than structure. This ensures that semantically similar content is grouped together.  

#### Tools:  
- `OllamaEmbeddings` or similar embedding models.  

#### Process:  
1. Compute embeddings for small sections of text.  
2. Merge sections with similar embeddings until a semantic threshold is reached.  

**Example**:  
```plaintext
Text: "Artificial intelligence is transforming industries. Machine learning is a subset of AI. Deep learning specializes in neural networks."
Chunks:  
1. "Artificial intelligence is transforming industries."  
2. "Machine learning is a subset of AI. Deep learning specializes in neural networks."
```



### 5. **Agentic Chunking**  
This advanced method uses AI to extract propositions or logical segments from text. It’s ideal for complex documents requiring logical segmentation.  

#### Tools:  
- `AgenticChunker` or similar frameworks.  

#### Example:  

**Input Text**:  
"Climate change is a global issue. Reducing emissions is crucial. Governments should enforce stricter laws."  

**Chunks**:  
1. "Climate change is a global issue."  
2. "Reducing emissions is crucial."  
3. "Governments should enforce stricter laws."  

**Use Case**: Extracting logical arguments for debate or analysis.


## Dependencies  

The project leverages the following Python libraries:  
- [LangChain](https://www.langchain.com): For text splitting and chunking.  
- [Rich](https://github.com/Textualize/rich): For enhanced console output.  
- [Chroma](https://www.trychroma.com): For vector database operations.  
- [LangChain Community Tools](https://github.com/langchain-community): For embeddings and specialized chunking methods.  


## Example Output  

Each chunking method demonstrates the segmented documents in the following format:  
```python
[Document(page_content='Chunk text here...', metadata={'Source': 'local'})]
```  
The semantic and agentic methods output chunks that align with the text's meaning or logical propositions.  

## Future Work  

- Add more document-specific chunkers for formats like HTML, JSON, and XML.  
- Evaluate the impact of chunking methods on RAG performance.  
- Integrate chunking with downstream NLP pipelines.  

## Contributing  
Contributions are welcome! Please submit a pull request or open an issue to discuss any improvements or suggestions.  
