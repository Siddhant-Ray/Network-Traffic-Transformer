FROM ubuntu:latest
LABEL MAINTAINER Siddhant Ray
WORKDIR /ntt
RUN set -xe \
    && apt-get update \
    && apt-get install python3-pip -y 

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .