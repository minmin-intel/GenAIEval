
from utils import get_test_data, get_args, get_doc_path
import requests
import time
import json
from docling.document_converter import DocumentConverter
import pandas as pd
import os
from utils import generate_answer
import glob
import random

def process_pdf_opea(args, doc_path):
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


def process_pdf_docling(doc_converter, doc_path):
    """
    doc_path: str
    """
    t0 = time.time()
    conv_res = doc_converter.convert(doc_path)
    t1= time.time()
    print(f"Time taken to process {doc_path}: {t1-t0}")

    # Export the full document to markdown
    full_doc = conv_res.document.export_to_markdown()
    return full_doc, conv_res

def get_context_of_table(table_md, full_doc):
    # locate the table in full doc
    # then get the sentences before the table.
    table_position = full_doc.find(table_md)
    if table_position != -1:
        context_start = max(0, table_position - 500)
        context_end = table_position
        context = full_doc[context_start:context_end]
        print(f"Context for table:\n{context}\n")
        return context
    else:
        print("Table not found in the document.")
        return ""


def get_table_summary_with_llm(table_md, args):
    """
    table_md: str
    args: including llm_endpoint_url and model 
    """
    prompt = f"You are a financial analyst. Read the table below and give it a descriptive title:\n{table_md} \nOnly output the table title."
    table_summary = generate_answer(args, prompt)
    print(f"Table summary:\n{table_summary}")
    return table_summary

def process_tables(conv_res, args):
    table_outputs = []
    for table_ix, table in enumerate(conv_res.document.tables):
        table_md = table.export_to_markdown()
        # get a description of table
        # context = get_context_of_table(table_md, full_doc) # bad performance
        context = get_table_summary_with_llm(table_md, args)
        table_outputs.append((table_md, context))
    return table_outputs        


def save_docling_output(full_doc, conv_res, output_dir):

    print("Length of full doc: ", len(full_doc))
    print("Number of tables: ", len(conv_res.document.tables))

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        doc_filename = conv_res.input.file.stem

        # Save the table as csv
        element_csv_filename = os.path.join(output_dir, f"{doc_filename}-table-{table_ix+1}.csv")
        table_df.to_csv(element_csv_filename)

        # Save the table as markdown
        element_md_filename = os.path.join(output_dir, f"{doc_filename}-table-{table_ix+1}.md")
        with open(element_md_filename, "w") as fp:
            fp.write(table.export_to_markdown())

    with open(os.path.join(output_dir,f"{conv_res.input.file.stem}.md"), "w") as fp:
        fp.write(full_doc)

from langchain_text_splitters import MarkdownTextSplitter
import os
def post_process_markdown(spliter: MarkdownTextSplitter, text: str) -> list:
    text = text.replace("## Table of Contents", "")
    text = text.replace("Table of Contents", "")
    return spliter.split_text(text)

from langchain_core.documents import Document
from uuid import uuid4
from langchain_chroma import Chroma

def index_chunks_into_chroma(embeddings, docs, metadata):
    """
    docs: list of str
    metadata: dict
    """
    vector_store = Chroma(
        collection_name="single_doc_collection",
        embedding_function=embeddings,
    )
    
    docs_to_add = []
    for doc in docs:
        doc = Document(
            page_content=doc,
            metadata=metadata,
        )
        docs_to_add.append(doc)
    
    uuids = [str(uuid4()) for _ in range(len(docs))]
    print("Adding documents to vector store........")
    t0 = time.time()
    vector_store.add_documents(documents=docs_to_add, ids=uuids)
    t1 = time.time()
    print(f"Time taken to add {len(docs)} documents: {t1-t0}")
    return vector_store


def index_tables_into_chroma(vector_store, tables, metadata):
    """
    tables: list of tuples (table_md, summary)
    metadata: dict

    embed the summary, but store table_md in metadata
    """
    
    docs_to_add = []
    for table in tables:
        doc = Document(
            page_content=table[1], # summary
            metadata={"table": table[0], **metadata},
        )
        docs_to_add.append(doc)
    
    uuids = [str(uuid4()) for _ in range(len(tables))]
    print("Adding tables to vector store........")
    t0 = time.time()
    vector_store.add_documents(documents=docs_to_add, ids=uuids)
    t1 = time.time()
    print(f"Time taken to add {len(tables)} tables: {t1-t0}")
    return vector_store

def single_doc_vectorstore(markdown_text, tables, metadata, embeddings):
    """
    markdown_text: str
    tables: list of tuples (table_md, summary)
    metadata: dict
    embeddings: HuggingFaceEmbeddings
    """
    spliter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunks = post_process_markdown(spliter, markdown_text)
    print(f"Number of chunks: {len(chunks)}")
    print("Average chunk size: ", sum([len(chunk) for chunk in chunks]) / len(chunks))
    print("Indexing text chunks into vector store........")
    t0 = time.time()
    vector_store = index_chunks_into_chroma(embeddings, chunks, metadata)
    t1 = time.time()
    print(f"Time taken to index {len(chunks)} chunks: {t1-t0}")

    print("Indexing tables into vector store........")
    t2 = time.time()
    vector_store = index_tables_into_chroma(vector_store, tables, metadata)
    t3 = time.time()
    print(f"Time taken to index {len(tables)} tables: {t3-t2}")

    print("Total time to index text and tables: ", t3-t0)
    return vector_store

def get_search_result(searched_doc):
    """
    searched_doc: Document
    """
    if "table" in searched_doc.metadata:
        print(f"@@@ Searched result is a table!")
        return searched_doc.metadata["table"]
    else:
        return searched_doc.page_content
    

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def post_process_text(text: str) -> str:
    text = text.replace("## Table of Contents", "")
    text = text.replace("Table of Contents", "")
    return text

def parent_child_retriever(doc, metadata, embeddings):
    """
    doc: one document
    Implement parent-child retriever
    """
    doc = post_process_text(doc)

    # This text splitter is used to create the parent documents
    parent_splitter = MarkdownTextSplitter(chunk_size=5000)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = MarkdownTextSplitter(chunk_size=1000)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="test_parent_child", embedding_function=embeddings
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    doc_to_add = Document(
        page_content=doc,
        metadata=metadata,
    )

    print("Starting to chunk and add document........")
    t0 = time.time()
    retriever.add_documents([doc_to_add])
    t1 = time.time()
    print(f"Time taken to chunk and add document: {t1-t0}")
    print("Number of parent documents: ", len(list(store.yield_keys())))
    return retriever, vectorstore


WORKDIR=os.getenv('WORKDIR')
DATAPATH=os.path.join(WORKDIR, 'datasets/financebench/dataprep/')

############### test table processing ##################
# if __name__ == "__main__":
#     args = get_args()
#     filename = os.path.join(DATAPATH, '3M_2018_10K.md')
#     with open(filename, 'r') as f:
#         full_doc = f.read()

#     full_doc = post_process_text(full_doc)    
#     table_mds = glob.glob(os.path.join(DATAPATH, "3M_2018_10K-table-*.md"))

#     tables = []
#     for table_md in random.sample(table_mds,5):
#         with open(table_md, "r") as f:
#             table_md = f.read()
#         print(f"Table markdown:\n{table_md}")
#         context = get_table_summary_with_llm(table_md, args)
#         tables.append((table_md, context))
#         print("="*50)
    


    

############### test chunking ###################

if __name__ == "__main__":
    args = get_args()
    filename = os.path.join(DATAPATH, '3M_2018_10K.md')
    with open(filename, 'r') as f:
        markdown_text = f.read()
    
    table_mds = glob.glob(os.path.join(DATAPATH, "3M_2018_10K-table-*.md"))

    tables = []
    t0 = time.time()
    for table_md in table_mds:
        with open(table_md, "r") as f:
            table_md = f.read()
        print(f"Table markdown:\n{table_md}")
        context = get_table_summary_with_llm(table_md, args)
        tables.append((table_md, context))
        print("="*50)
    t1 = time.time()
    print(f"Time taken to process {len(table_mds)} tables: {t1-t0}")
    print(f"Average time taken to process a table: {(t1-t0)/len(table_mds)}")

    metadata = {"company": "3M", "year": 2018}

    model = "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model)

    if args.retriever_option == "parent_child":
        print("Using parent-child retriever")
        retriever, vectorstore = parent_child_retriever(markdown_text, metadata, embeddings)
    elif args.retriever_option == "plain":
        print("Using plain retriever")
        vectorstore = single_doc_vectorstore(markdown_text, tables, metadata, embeddings)
    else:
        raise ValueError("Invalid retriever option. Please choose either 'parent_child' or 'plain'.")


    df = get_test_data()
    df = df.loc[df["doc_name"]=="3M_2018_10K"]

    for _, row in df.iterrows():
        query = row["question"]
        print(f"Query: {query}")
        if args.retriever_option == "parent_child":
            child_doc = vectorstore.similarity_search(
                query,
                k=1,
            )
            print(f"Number of child documents: {len(child_doc)}")
            print(f"Child doc length: {len(child_doc[0].page_content)}")

            results = retriever.invoke(query, k=1)
            print(f"Number of parent documents: {len(results)}")
            for res in results:
                print(f"Parent doc length: {len(res.page_content)}")
                print(f"* {res.page_content}")
        elif args.retriever_option == "plain":
            results = vectorstore.similarity_search(
                query,
                k=1,
            )
            print(f"Number of documents: {len(results)}")
            for res in results:
                content = get_search_result(res)
                print(f"Doc length: {len(content)}")
                print(f"* {content}")
        print("="*50)



################## test pdf parsing with docling ###################
# if __name__ == "__main__":
#     args = get_args()
#     df = get_test_data()

#     df = df.loc[df["company"]=="3M"]
#     print("There are {} questions to be answered.".format(df.shape[0]))
    
#     docs = df["doc_name"].unique().tolist()
#     doc_paths = [get_doc_path(doc) for doc in docs]
#     print(f"There are {len(doc_paths)} unique documents to be processed.")

#     doc_paths = [doc_paths[0]]

#     t0 = time.time()

#     if args.ingest_option == "docling":
#         doc_converter = DocumentConverter()
#         for doc_path in doc_paths:
#             full_doc, conv_res=process_pdf_docling(doc_converter, doc_path)
#             output_dir = args.output
#             save_docling_output(full_doc, conv_res, output_dir)
#     elif args.ingest_option == "opea":
#         process_pdf_opea(args, doc_paths)
#     else:
#         raise ValueError("Invalid ingest option. Please choose either 'docling' or 'opea'.")
    
#     t1 = time.time()
#     print(f"Time taken to process {len(doc_paths)} PDF with {args.ingest_option}: {t1-t0}")
#     print(f"Average time taken to process a PDF: {(t1-t0)/len(doc_paths)}")
