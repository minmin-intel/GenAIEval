from ingest_data import process_pdf_docling
from transformers import AutoTokenizer
import os
import json
import time
from docling.document_converter import DocumentConverter

from prompt import LONG_CONTEXT_PROMPT_TEMPLATE
from utils import generate_answer, get_test_data, get_args, get_doc_path


MAX_CONTEXT_LEN = 131072
BUFFER = 100

def truncate_context(full_doc, question, tokenizer, max_output_tokens):
    prompt = LONG_CONTEXT_PROMPT_TEMPLATE.format(document=full_doc, question=question)
    prompt_len = len(tokenizer.encode(prompt))
    print(f"Prompt length in tokens: {prompt_len}")
    
    max_input_tokens = MAX_CONTEXT_LEN - BUFFER - max_output_tokens
    if prompt_len > max_input_tokens:
        overflow = prompt_len - max_input_tokens
        print(f"Overflow: {overflow}")
        full_doc_tokens = tokenizer.encode(full_doc)
        truncated_doc_tokens = full_doc_tokens[:-overflow]
        truncated_doc = tokenizer.decode(truncated_doc_tokens)
        print(f"Truncated doc from {len(full_doc_tokens)} to {len(truncated_doc_tokens)}")
        print(truncated_doc[:50])
        # prompt = LONG_CONTEXT_PROMPT_TEMPLATE.format(document=truncated_doc, question=question)
        # prompt_len = len(tokenizer.encode(prompt))
        # print(f"New prompt length in tokens: {prompt_len}")
        return truncated_doc
    else:
        return full_doc


WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

if __name__ == "__main__":
    args = get_args()

    df = get_test_data()
    df = df.loc[df["doc_name"]!="3M_2018_10K"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.debug:
        filename = os.path.join(DATAPATH, '3M_2018_10K.md')
        with open(filename, "r") as f:
            full_doc = f.read()
    else:
        doc_converter = DocumentConverter()

    previous_doc_name = ""

    responses = []
    truncated_flags = []
    generation_time = []
    for i, row in df.iterrows():
        query = row["question"]
        doc_name = row["doc_name"]

        print(f"Question: {query}\nDocument: {doc_name}")

        if doc_name != previous_doc_name:
            print(f" @@@ Question is about New document: {doc_name}")

            if not args.debug:
                doc_path = get_doc_path(doc_name)
                print("Parsing PDF....")
                full_doc, _ = process_pdf_docling(doc_converter, doc_path)
                doc_save_path = os.path.join(DATAPATH, f"{doc_name}.md")
                with open(doc_save_path, "w") as f:
                    f.write(full_doc)

            previous_doc_name = doc_name
            truncated_doc = truncate_context(full_doc, query, tokenizer, args.max_new_tokens)
        else:
            print(f" @@@ Question is about the same document: {doc_name}")


        if len(truncated_doc) < len(full_doc):
            truncated_flags.append("true")
        else:
            truncated_flags.append("false")

        prompt = LONG_CONTEXT_PROMPT_TEMPLATE.format(document=truncated_doc, question=query)

        t0 = time.time()
        resp = generate_answer(args, prompt)
        t1 = time.time()
        print(f"Response: {resp}")

        responses.append(resp)
        generation_time.append(t1-t0)

        output = {
            "doc_name": doc_name,
            "question": query,
            "gold_answer": row["answer"],
            "response": resp,
            "truncated": truncated_flags[-1],
            "generation_time": generation_time[-1]
        }

        with open(args.output, "a") as f:
            f.write(json.dumps(output)+"\n")

        print("="*50)

    df["response"] = responses
    df["truncated"] = truncated_flags
    df["generation_time"] = generation_time
    df.to_csv(args.output.replace(".json", ".csv"), index=False)

    