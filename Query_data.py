from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import argparse
# Python lib to parse command-line arguments
# Allow users to input directly from terminal and run the code
from langchain.prompts import PromptTemplate
from transformers import pipeline
# simplifies the use of pre-trained models for various tasks, including question answering

Chroma_path = 'chroma' # Path where chunks are saved

# Define prompt template for QA model
prompt_template = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Command-Line Interface (CLI): users to interact with a computer program by typing commands into a console or terminal window
    parser =argparse.ArgumentParser()
    # Initialize a new arg parser to handle input
    parser.add_argument('query_text', type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Initialize the Chroma Database
    embeddings_func = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory = Chroma_path,
        embedding_function =embeddings_func
    )

    # Search database for relevant chunks
    results = db.similarity_search_with_score(query_text, k =5)

    filtered_results = [doc for doc, score in results if score >= 0.6]
    if not filtered_results:
        print('No matching results above the relevance threshold.')
        return

    # Combine retrieved documents into context
    context_text = "\n\n---\n\n".join([doc.page_content for doc in filtered_results])
    #  combines the page content of the retrieved documents into a single string, separated by a delimiter (\n\n---\n\n)
    #  This combined context will be used as input for the question-answering model.

    # Format the prompt
    prompt_format = PromptTemplate.from_template(prompt_template)
    prompt = prompt_format.format(context=context_text, question=query_text)

    # Initiate the QA Model
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    # Generate answers
    response = qa_model(question = query_text, context = context_text)

    # Extract and Display Sources
    sources = list(set([doc.metadata.get("source", None) for doc in filtered_results]))
    # Extracts the metadata (source) for each document retrieved from the database
    # Trace back where the information came from.
    formatted_response = f"{prompt}\n\nAnswer: {response['answer']}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    # ensures that certain code only runs when the script is executed directly and not when imported as a module
    main()
