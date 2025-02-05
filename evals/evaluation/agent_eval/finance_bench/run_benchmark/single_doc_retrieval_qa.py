from utils import generate_answer, get_test_data, get_args, get_doc_path
from prompt import RAG_PROMPT_TEMPLATE
import requests
import time
import json


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


if __name__ == "__main__":
    args = get_args()
    df = get_test_data()

    df = df.loc[df["company"]=="3M"]
    print("There are {} questions to be answered".format(df.shape[0]))

    for i, row in df.iterrows():
        doc_name = row["doc_name"]
        # if doc_name not in docs:
        #     doc_path = get_doc_path(doc_name)
        #     t0 = time.time()
        #     process_pdf(args, doc_path)
        #     t1 = time.time()
        #     print(f"Time taken to process PDF: {t1-t0}")
        #     docs.append(doc_name)
        #     process_time.append(t1-t0)

        query = row["question"]
        print("Query:\n", query)
        url = args.retrieval_endpoint_url
        context = search_knowledge_base(url, query)
        print("Retrieved context:\n", context)
        print("-"*50)
        print("Oracle evidence:\n", row["evidence"])
        print("-"*50)


        # prompt = RAG_PROMPT_TEMPLATE.format(document=context, question=query)
        # resp = generate_answer(args.prompt)
        # print("RAG Answer:\n", resp)
        # print("-"*50)
        # print("Gold Answer:\n", row["answer"])
        # print("="*100)
        
        output = {
            "doc_name": doc_name,
            "question": query,
            "context": context,
            # "response": resp,
            # "gold_answer": row["answer"],
            "oracle_evidence": row["evidence"],
        }

        with open(args.output, "a") as f:
            f.write(json.dumps(output)+"\n")


