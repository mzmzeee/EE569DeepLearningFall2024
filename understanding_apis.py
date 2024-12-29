import os
from openai import OpenAI
import tiktoken
from pprint import pprint

if os.getenv("OPENAI_API_KEY") is None:
  with open('key.txt', 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.readline().strip()

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")

encoding = tiktoken.encoding_for_model("text-davinci-003")
enc = encoding.encode("Weights & Biases is awesome!")
print(enc)
print(encoding.decode(enc))

for token_id in enc:
    print(f"{token_id}\t{encoding.decode([token_id])}")

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
enc = encoding.encode("Weights & Biases is awesome!")
print(enc)
print(encoding.decode(enc))

for token_id in enc:
    print(f"{token_id}\t{encoding.decode([token_id])}")

client = OpenAI()

def generate_with_temperature(temp):
  "Generate text with a given temperature, higher temperature means more randomness"
  completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {
              "role": "user",
              "content": "complete the story: once opon a time, there was a rabbit"
          }
      ],
      max_completion_tokens=100,
      temperature=temp,
  )
  return completion.choices[0].message.content

for temp in [0, 0.5, 1, 1.5, 2]:
  pprint(f'TEMP: {temp}, GENERATION: {generate_with_temperature(temp)}')

def generate_with_topp(top_p):
  "Generate text with a given temperature, higher temperature means more randomness"
  completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {
              "role": "user",
              "content": "complete the story: once opon a time, there was a rabit"
          }
      ],
      max_completion_tokens=100,
      top_p=top_p,
  )
  return completion.choices[0].message.content


for topp in [0.01, 0.1, 0.5, 1]:
  pprint(f'TOP_P: {topp}, GENERATION: {generate_with_topp(topp)}')