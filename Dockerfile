FROM python:3.9.1
ADD . .
WORKDIR .
RUN pip install -r requirements.txt
RUN python main.py