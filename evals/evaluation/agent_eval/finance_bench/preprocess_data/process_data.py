import pandas as pd
import os

WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'financebench/data/')
filename = "financebench_open_source.jsonl"

df = pd.read_json(DATAPATH + filename, lines=True)

for _, row in df.iterrows():
    doc_name = row["doc_name"]
    if "," in doc_name:
        print("Detected more than one doc_name in a single row")
        print(doc_name)
