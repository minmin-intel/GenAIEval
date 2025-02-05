from openai import OpenAI
import pandas as pd
import os
import argparse

"""
    chunk_size: int = Form(1500),
    chunk_overlap: int = Form(100),
    process_table: bool = Form(False),
    table_strategy: str = Form("fast"),
"""
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_endpoint_url", type=str, default="http://localhost:8086")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--ip_address", type=str, default="localhost")
    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--retrieval_endpoint_url", type=str, default="http://localhost:8889/v1/retrievaltool")
    parser.add_argument("--output", type=str, default="output.jsonl")
    args = parser.parse_args()
    return args

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'financebench/data/')
PDFPATH=os.path.join(WORKDIR, 'financebench/pdfs/')

def get_test_data():
    filename = "financebench_open_source.jsonl"
    df = pd.read_json(DATAPATH + filename, lines=True)
    return df

def get_doc_path(doc_name):
    return os.path.join(PDFPATH, doc_name)+".pdf"

def generate_answer(args, prompt):
    """
    Use vllm endpoint to generate the answer
    """

    # assemble prompt with parsed document and question

    # send request to vllm endpoint
    client = OpenAI(
        base_url=f"{args.endpoint_url}/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    # get response
    response = completion.choices[0].message.content

    return response

