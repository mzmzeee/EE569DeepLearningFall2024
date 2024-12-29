import os
import tiktoken
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# Configure OpenAI API Key
def configure_openai_api_key():
  if os.getenv("OPENAI_API_KEY") is None:
    with open('key.txt', 'r') as f:
      os.environ["OPENAI_API_KEY"] = f.readline().strip()
  assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "Invalid OpenAI API key"
  print("OpenAI API key configured")

configure_openai_api_key()

# Define Model Constants
MODEL_NAME = "gpt-4o-mini"

# Load Documents
dl = DirectoryLoader("docs_sample", "**/*.md")
documents = dl.load()
print(f"Loaded {len(documents)} documents.")

# Tokenizer for counting tokens
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

# Function to count tokens in each document
def count_tokens(documents):
  return [len(tokenizer.encode(doc.page_content)) for doc in documents]

token_counts = count_tokens(documents)
print(f"Token counts per document: {token_counts}")

# Split Documents into Sections
md_text_splitter = MarkdownTextSplitter(chunk_size=1000)
document_sections = md_text_splitter.split_documents(documents)
print(f"Total document sections: {len(document_sections)}")
print(f"Maximum tokens in a section: {max(count_tokens(document_sections))}")
print(f"Example document section content:\n{document_sections[0].page_content}")

# Text Embedding and Retrieval
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(document_sections, embeddings)
retriever = db.as_retriever(search_kwargs=dict(k=3))  # Retrieve top 3 results

# Sample Query and Retrieval
query = "How can I share my W&B report with my team members in a public W&B project?"
docs = retriever.invoke(query)

print("Relevant documents retrieved:")
for doc in docs:
  print(f"- {doc.metadata['source']}")

# Generate Prompt using Retrieved Documents
prompt_template = """
Use the following information to answer the question at the end:

{context}

Question: {question}
Helpful Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

context = "\n\n".join([doc.page_content for doc in docs])
prompt = PROMPT.format(context=context, question=query)

print("Generated Prompt:")
print(prompt)

# Answer the Query using LLM
llm = OpenAI()
response = llm.invoke(prompt)
print("LLM Response:")
print(response)

# Combine Retrieval and LLM for Comprehensive Answer
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
result = qa.invoke(query)

print("Final Answer with Retrieved Context:")
print(result)