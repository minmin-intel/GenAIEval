from utils import generate_answer, get_test_data, get_args, run_agent, get_doc_path
from prompt import RAG_PROMPT_TEMPLATE
import requests
import time
import json
from ingest_data import process_pdf_docling
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from docling.document_converter import DocumentConverter
from ingest_data import single_doc_vectorstore


def search_knowledge_base(url, query):
    """
    Use OPEA DocIndexRetriever to retrieve relevant context
    """
    print(url)
    proxies = {"http": ""}
    payload = {
        "input": query,
    }
    response = requests.post(url, json=payload, proxies=proxies)
    print(response)
    if "documents" in response.json():
        docs = response.json()["documents"]
        context = ""
        for i, doc in enumerate(docs):
            if i == 0:
                context = doc
            else:
                context += "\n" + doc
        # print(context)
        return context
    elif "text" in response.json():
        return response.json()["text"]
    elif "reranked_docs" in response.json():
        docs = response.json()["reranked_docs"]
        context = ""
        for i, doc in enumerate(docs):
            if i == 0:
                context = doc["text"]
            else:
                context += "\n" + doc["text"]
        # print(context)
        return context
    else:
        return "Error parsing response from the knowledge base."

import os
if __name__ == "__main__":
    WORKDIR=os.getenv('WORKDIR')
    DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')
    args = get_args()
    df = get_test_data()

    df = df.loc[df["company"]=="3M"]
    # df = df.loc[df["doc_name"]=="3M_2018_10K"]
    print("There are {} questions to be answered".format(df.shape[0]))

    docs = []
    process_time = []

    model = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model)

    doc_converter = DocumentConverter()
    
    for i, row in df.iterrows():
        doc_name = row["doc_name"]
        if doc_name not in docs:
            doc_path = get_doc_path(doc_name)
            t0 = time.time()
            # # convert pdf to markdown with docling
            text,_=process_pdf_docling(doc_converter, doc_path)
            t1 = time.time()
            # print(f"Time taken to process PDF: {t1-t0}")
            # filename = os.path.join(DATAPATH, '3M_2018_10K.md')
            # with open(filename, 'r') as f:
            #     text = f.read()
            
            # index into vectorstore
            vectorstore = single_doc_vectorstore(text, {"company": "3M", "year": 2018}, embeddings)
            t2 = time.time()
            print(f"Time taken to index into vectorstore: {t2-t1}")
            
            docs.append(doc_name)
            process_time.append(t2-t0)


        
        query = row["question"]
        print("Query:\n", query)
        # url = args.retrieval_endpoint_url
        # context = search_knowledge_base(url, query)
        results = vectorstore.similarity_search(query, k=1)
        context = results[0].page_content
        # print("Retrieved context:\n", context)
        print("-"*50)

        # Conventional RAG
        prompt = RAG_PROMPT_TEMPLATE.format(document=context, question=query)
        resp = generate_answer(args, prompt)
        print("RAG Answer:\n", resp)
        print("-"*50)

        # RAG agent
        agent_resp = run_agent(args, query)
        print("Agent Answer:\n", agent_resp)
        print("-"*50)

        # print("Gold Answer:\n", row["answer"])
        print("="*100)
        
        output = {
            "doc_name": doc_name,
            "question": query,
            "rag_response": resp,
            "agent_response": agent_resp,
            "gold_answer": row["answer"],
            "oracle_evidence": row["evidence"],
            "context": context,
        }

        with open(args.output, "a") as f:
            f.write(json.dumps(output)+"\n")


