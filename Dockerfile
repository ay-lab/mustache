FROM continuumio/anaconda3

RUN apt update && apt install build-essential zlib1g-dev

RUN git clone https://github.com/ay-lab/mustache

RUN conda env create -f /mustache/environment.yml

RUN conda activate mustache