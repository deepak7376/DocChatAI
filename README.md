# DocChatAI | LLM-based Custom PDF Chatbot

[Project link](https://huggingface.co/spaces/Deepak7376/LLM-based-custom-pdf-chatbot)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

LLM-based Custom PDF Chatbot is a chatbot application that utilizes a Language Model (LLM) to process and interact with custom PDF files. The chatbot is designed to extract information, answer questions, and provide assistance based on the content of PDF documents.


## How It Works

1. **Indexing:**
    - **Pipeline:** A pipeline for ingesting data from a source and indexing it. This usually happens offline.
    - **Common Sequence:**
        - **Load:** First, we need to load our data. We’ll use DocumentLoaders for this.
        - **Split:** Text splitters break large Documents into smaller chunks. This is useful both for indexing data and for passing it into a model, since large chunks are harder to search over and won’t fit in a model’s finite context window.
        - **Store:** We need somewhere to store and index our splits so that they can later be searched over. This is often done using a VectorStore and Embeddings model.
2. **Retrieval and Generation:**
    - **RAG Chain:** The actual RAG chain takes the user query at runtime and retrieves the relevant data from the index, then passes that to the model.
    - **Common Sequence:**
        - **Retrieve:** Given a user input, relevant splits are retrieved from storage using a Retriever.
        - **Generate:** A ChatModel / LLM produces an answer using a prompt that includes the question and the retrieved data.

This project was developed as an entry for the Streamlit Hackathon in September 2023.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/deepak7376/DocChatAI
   ```

2. Navigate to the project directory:

   ```bash
   cd DocChatAI
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:8000` to interact with the chatbot.


## Connect with Me

- 🌐 Website: [Portfolio](http://deepak7376.github.io/)
- 💬 Discord: [Join the Community](https://discord.gg/community)
- 💼 LinkedIn: [Deepak Yadav](https://www.linkedin.com/in/dky7376/)

## License

This project is licensed under the [MIT License](LICENSE).

