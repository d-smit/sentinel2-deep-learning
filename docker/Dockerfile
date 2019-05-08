FROM python:3.7

RUN apt update

RUN mkdir app data
#COPY requirements.txt /app/

COPY . /app
WORKDIR /app

RUN pip install -r /app/requirements.txt
