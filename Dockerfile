FROM python:3.10-slim-buster
WORKDIR /nursery
COPY . /nursery
RUN pip install -r requirements.txt
CMD ["python", "main.py"]