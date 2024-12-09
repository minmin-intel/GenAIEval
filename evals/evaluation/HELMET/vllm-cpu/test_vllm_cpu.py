from openai import OpenAI

endpoint_url = "http://localhost:8000/v1"
print(f"** Endpoint URL: {endpoint_url}")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = OpenAI(
        base_url=endpoint_url,
        api_key="EMPTY_KEY",
    )

query = "What is deep learning?" * 200000


completion = model.chat.completions.create(
  model=model_id,
  messages=[
    {"role": "user", "content": query}
  ],
#   max_tokens=20,
  temperature = 0.0,
)

print(completion.choices[0].message.content)