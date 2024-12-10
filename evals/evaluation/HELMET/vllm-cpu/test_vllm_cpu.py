from openai import OpenAI
import os

host_ip = os.getenv("host_ip", "localhost")
port=8085

endpoint_url = f"http://{host_ip}:{port}/v1"
print(f"** Endpoint URL: {endpoint_url}")

model_id = "meta-llama/Llama-3.2-1B-Instruct" #"meta-llama/Meta-Llama-3.1-8B-Instruct"

model = OpenAI(
        base_url=endpoint_url,
        api_key="EMPTY_KEY",
    )

query = "What is deep learning?"


completion = model.chat.completions.create(
  model=model_id,
  messages=[
    {"role": "user", "content": query}
  ],
  max_tokens=20,
  temperature = 0.0,
)

print(completion.choices[0].message.content)