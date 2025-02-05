
from utils import get_test_data, get_args, get_doc_path
import requests
import time
import json

def process_pdf(args, doc_path):
    """
    Use OPEA dataprep microservice to parse the PDF document
    doc_path: str, or list
    """
    # ingest api
    url=f"http://{args.ip_address}:6007/v1/dataprep/ingest"
    proxies = {"http": ""}
    if isinstance(doc_path, str):
        doc_path = [doc_path]
    files = [("files", (f, open(f, "rb"))) for f in doc_path]
    payload = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "process_table": "true", "table_strategy":"hq"}
    resp = requests.request("POST", url=url, headers={}, files=files, data=payload, proxies=proxies)
    print(resp)


if __name__ == "__main__":
    args = get_args()
    df = get_test_data()

    df = df.loc[df["company"]=="3M"]
    print("There are {} questions to be answered.".format(df.shape[0]))
    
    docs = df["doc_name"].unique().tolist()
    doc_paths = [get_doc_path(doc) for doc in docs]
    print(f"There are {len(doc_paths)} unique documents to be processed.")

    t0 = time.time()
    process_pdf(args, doc_paths)
    t1 = time.time()
    print(f"Time taken to process {len(doc_paths)} PDF: {t1-t0}")
    print(f"Average time taken to process a PDF: {(t1-t0)/len(doc_paths)}")