FROM amazon/aws-lambda-python:3.7
#FROM python:3.7-slim

COPY requirements.txt .
RUN python3.7 -m pip install -r requirements.txt -t .

COPY . . 

CMD ["app.tag_invoice"]
# ENTRYPOINT ["python", "app.py"]
