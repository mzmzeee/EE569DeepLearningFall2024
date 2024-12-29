import os
import random
import shutil
import subprocess

from openai import OpenAI
import tiktoken

from pathlib import Path

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential, # for exponential backoff
)
import wandb
from wandb.integration.openai import autolog


doc_list =["collaborate-on-reports.md",
           "dataset-versioning.md",
           "getting-started.md",
           "lightning.md",
           "model-management-concepts.md",
           "quickstart.md",
           "tables-quickstart.md",
           "tags.md",
           "teams.md",
           "track-external-files.md",
           "walkthrough.md"]

# Download files
if not Path("examples.txt").exists():
    subprocess.run("wget https://raw.githubusercontent.com/wandb/edu/main/llm-apps-course/notebooks/examples.txt", shell=True)
if not Path("prompt_template.txt").exists():
    subprocess.run("wget https://raw.githubusercontent.com/wandb/edu/main/llm-apps-course/notebooks/prompt_template.txt", shell=True)
if not Path("docs_sample/").exists():
    os.mkdir("docs_sample")
    for doc in doc_list:
        subprocess.run("wget https://raw.githubusercontent.com/wandb/edu/main/llm-apps-course/docs_sample/" + doc, shell=True)
        shutil.move(doc, "docs_sample")

if os.getenv("OPENAI_API_KEY") is None:
  with open('key.txt', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.readline().strip()

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

client = OpenAI()

os.environ["WANDB_PROJECT"] = "llmapps"
wandb.init()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

MODEL_NAME = "gpt-4o-mini"

system_prompt = "You are a helpful assistant."
user_prompt = "Generate a support question from a W&B user"


def generate_and_print(system_prompt, user_prompt, n=5):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    responses = completion_with_backoff(
        model=MODEL_NAME,
        messages=messages,
        n=n,
    )
    for response in responses.choices:
        generation = response.message.content
        print(generation)


generate_and_print(system_prompt, user_prompt)


delimiter = "\t" # tab separated queries
with open("examples.txt", "r") as file:
    data = file.read()
    real_queries = data.split(delimiter)

print(f"We have {len(real_queries)} real queries:")
print(f"Sample one: \n\"{random.choice(real_queries)}\"")

def generate_few_shot_prompt(queries, n=3):
    prompt = "Generate a support question from a W&B user\n" +\
        "Below you will find a few examples of real user queries:\n"
    for _ in range(n):
        prompt += random.choice(queries) + "\n"
    prompt += "Let's start!"
    return prompt

generation_prompt = generate_few_shot_prompt(real_queries)
print(generation_prompt)

generate_and_print(system_prompt, user_prompt=generation_prompt)

def find_md_files(directory):
    "Find all markdown files in a directory and return their content and path"
    md_files = []
    for file in Path(directory).rglob("*.md"):
        with open(file, 'r', encoding='utf-8') as md_file:
            content = md_file.read()
        md_files.append((file.relative_to(directory), content))
    return md_files

documents = find_md_files('docs_sample/')
print(f"number of documents:{len(documents)}")

tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
tokens_per_document = [len(tokenizer.encode(document)) for _, document in documents]
print(tokens_per_document)


# extract a random chunk from a document
def extract_random_chunk(document, max_tokens=512):
    tokens = tokenizer.encode(document)
    if len(tokens) <= max_tokens:
        return document
    start = random.randint(0, len(tokens) - max_tokens)
    end = start + max_tokens
    return tokenizer.decode(tokens[start:end])

def generate_context_prompt(chunk):
    prompt = "Generate a support question from a W&B user\n" +\
        "The question should be answerable by provided fragment of W&B documentation.\n" +\
        "Below you will find a fragment of W&B documentation:\n" +\
        chunk + "\n" +\
        "Let's start!"
    return prompt

chunk = extract_random_chunk(documents[0][1])
generation_prompt = generate_context_prompt(chunk)

print(generation_prompt)

generate_and_print(system_prompt, generation_prompt, n=3)

# read system_template.txt file into an f-string
with open("system_template.txt", "r") as file:
    system_prompt = file.read()

# read prompt_template.txt file into an f-string
with open("prompt_template.txt", "r") as file:
    prompt_template = file.read()

print(system_prompt)
print(prompt_template)

def generate_context_prompt(chunk, n_questions=3):
    questions = '\n'.join(random.sample(real_queries, n_questions))
    user_prompt = prompt_template.format(QUESTIONS=questions, CHUNK=chunk)
    return user_prompt

user_prompt = generate_context_prompt(chunk)

print(user_prompt)

def generate_questions(documents, n_questions=3, n_generations=5):
    questions = []
    for _, document in documents:
        chunk = extract_random_chunk(document)
        user_prompt = generate_context_prompt(chunk, n_questions)
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = completion_with_backoff(
            model=MODEL_NAME,
            messages=messages,
            n = n_generations,
            )
        questions.extend([response.choices[i].message.content for i in range(n_generations)])
    return questions


# function to parse model generation and extract CONTEXT, QUESTION and ANSWER
def parse_generation(generation):
    lines = generation.split("\n")
    context = []
    question = []
    answer = []
    flag = None

    for line in lines:
        if "CONTEXT:" in line:
            flag = "context"
            line = line.replace("CONTEXT:", "").strip()
        elif "QUESTION:" in line:
            flag = "question"
            line = line.replace("QUESTION:", "").strip()
        elif "ANSWER:" in line:
            flag = "answer"
            line = line.replace("ANSWER:", "").strip()

        if flag == "context":
            context.append(line)
        elif flag == "question":
            question.append(line)
        elif flag == "answer":
            answer.append(line)

    context = "\n".join(context)
    question = "\n".join(question)
    answer = "\n".join(answer)
    return context, question, answer


generations = generate_questions([documents[0]], n_questions=3, n_generations=5)
parse_generation(generations[0])

parsed_generations = []
generations = generate_questions(documents, n_questions=3, n_generations=5)
for generation in generations:
    context, question, answer = parse_generation(generation)
    parsed_generations.append({"context": context, "question": question, "answer": answer})

# let's convert parsed_generations to a pandas dataframe and save it locally
df = pd.DataFrame(parsed_generations)
df.to_csv('generated_examples.csv', index=False)

# log df as a table to W&B for interactive exploration
wandb.log({"generated_examples": wandb.Table(dataframe=df)})

# log csv file as an artifact to W&B for later use
artifact = wandb.Artifact("generated_examples", type="dataset")
artifact.add_file("generated_examples.csv")
wandb.log_artifact(artifact)
wandb.finish()
