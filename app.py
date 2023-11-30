import base64
import os

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from streamlit_chat import message
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

st.set_page_config(layout="wide")

def process_answer(instruction, qa_chain):
    response = ''
    generated_text = qa_chain.run(instruction)
    return generated_text

# Function to read the current visitor count from a file
def read_visitor_count():
    try:
        with open("visitor_count.txt", "r") as file:
            count = int(file.read())
    except FileNotFoundError:
        count = 0
    return count

# Function to update and write the visitor count to a file
def update_visitor_count():
    count = read_visitor_count() + 1
    with open("visitor_count.txt", "w") as file:
        file.write(str(count))
    return count

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)
    vectordb.save_local("faiss_index")


@st.cache_resource
def initialize_qa_chain(selected_model):
    # Constants
    CHECKPOINT = selected_model
    TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
    BASE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT, device_map=torch.device('cpu'), torch_dtype=torch.float32)
    pipe = pipeline(
        'text2text-generation',
        model=BASE_MODEL,
        tokenizer=TOKENIZER,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        # device=torch.device('cpu')
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = FAISS.load_local("faiss_index", embeddings)

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return qa_chain


@st.cache_data
# function to display the PDF of a given file
def display_pdf(file):
    try:
        # Opening file from file path
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # Embedding PDF in HTML
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

        # Displaying File
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while displaying the PDF: {e}")


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], key=str(i))


def main():
    # Update the visitor count and get the updated count
    visitor_count = update_visitor_count()

    # Display the current visitor count
    st.write(f"Total Visitors: {visitor_count}")
     # Add a sidebar for model selection
    model_options = ["MBZUAI/LaMini-T5-738M", "google/flan-t5-base", "google/flan-t5-small"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    
    st.markdown("<h1 style='text-align: center; color: blue;'>Custom PDF Chatbot 🦜📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF, and Ask Questions 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])
    
    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        os.makedirs("docs", exist_ok=True)
        filepath = os.path.join("docs", uploaded_file.name)
        try:
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
                st.json(file_details)
                st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
                pdf_view = display_pdf(uploaded_file)

            with col2:
                st.success(f'model selected successfully: {selected_model}')
                with st.spinner('Embeddings are in process...'):
                    ingested_data = data_ingestion()
                st.success('Embeddings are created successfully!')
                st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

                user_input = st.text_input("", key="input")

                # Initialize session state for generated responses and past messages
                if "generated" not in st.session_state:
                    st.session_state["generated"] = ["I am ready to help you"]
                if "past" not in st.session_state:
                    st.session_state["past"] = ["Hey there!"]

                # Search the database for a response based on user input and update session state
                if user_input:
                    answer = process_answer({'query': user_input}, initialize_qa_chain(selected_model))
                    st.session_state["past"].append(user_input)
                    response = answer
                    st.session_state["generated"].append(response)

                # Display conversation history using Streamlit messages
                if st.session_state["generated"]:
                    display_conversation(st.session_state)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
