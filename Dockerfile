FROM python:3.10-slim

WORKDIR /app

COPY train.py .
COPY predict.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN python train.py

CMD ["python", "predict.py"]
