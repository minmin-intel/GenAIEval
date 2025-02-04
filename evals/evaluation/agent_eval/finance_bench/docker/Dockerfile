FROM ubuntu:22.04

WORKDIR /home/user

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10  \
    python3-pip \
    tesseract-ocr \
    libpoppler-cpp-dev \
    wget \
    git \
    poppler-utils \
    libmkl-dev \
    curl \
    nano


COPY requirements.txt /home/user/requirements.txt

RUN pip install -r requirements.txt

WORKDIR /home/user