import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import pytesseract
import cv2
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pandas as pd
import os
from openai import OpenAI
import argparse
from .prompt import LONG_CONTEXT_PROMPT_TEMPLATE
from .utils import generate_answer, get_test_data, get_args


def process_page(doc, idx):
    page = doc.load_page(idx)
    pagetext = page.get_text().strip()
    result = pagetext if pagetext.endswith(("!", "?", ".")) else pagetext + "."

    page_images = doc.get_page_images(idx)
    if page_images:
        for img_index, img in enumerate(page_images):
            xref = img[0]
            img_data = doc.extract_image(xref)
            img_bytes = img_data["image"]

            # process images
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_result = pytesseract.image_to_string(img_array, lang="eng", config="--psm 6")

            # add results
            pageimg = img_result.strip()
            pageimg += "" if pageimg.endswith(("!", "?", ".")) else "."
            result += pageimg
    return result


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_page, doc, i) for i in range(doc.page_count)]
        for future in as_completed(futures):
            results.append(future.result())

    combined_result = "".join(results)
    return combined_result

def get_separators():
    separators = [
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ]
    return separators

def split_text(content, text_splitter):
    chunks = text_splitter.split_text(content)
    return chunks


def parse_pdf_document(doc_name, chunk_size=1000, chunk_overlap=100):
    """
    Use OPEA dataprep microservice to parse the PDF document
    """
    # load pdf
    pdf_content = load_pdf(doc_name)

    # text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #         add_start_index=True,
    #         separators=get_separators(),
    #     )
    
    # chunks = split_text(pdf_content, text_splitter)

    return pdf_content



if __name__ == "__main__":
    args = get_args()

    df = get_test_data()
    doc_name = df.iloc[0]["doc_name"]
    print(doc_name)
    parsed_content = parse_pdf_document(doc_name)
    print(parsed_content)
    # generate_answer()

    