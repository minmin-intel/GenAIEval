from utils import generate_answer, get_test_data, get_args, get_doc_path
import requests

def process_pdf(args, doc_path):
    """
    Use OPEA dataprep microservice to parse the PDF document
    """
    # ingest api
    url=f"http://{args.ip_address}:6007/v1/dataprep/ingest"
    proxies = {"http": ""}
    files = [("files", (f, open(f, "rb"))) for f in [doc_path]]
    payload = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "process_table": "true"}
    resp = requests.request("POST", url=url, headers={}, files=files, data=payload, proxies=proxies)
    print(resp)

if __name__ == "__main__":
    args = get_args()
    df = get_test_data()
    doc_name = df.iloc[0]["doc_name"]
    print(doc_name)
    doc_path = get_doc_path(doc_name)
    process_pdf(args, doc_path)
    # generate_answer()


