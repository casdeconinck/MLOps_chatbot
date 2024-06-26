FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "-m", "chainlit", "run", "app.py", "--port", "8080", "--host", "0.0.0.0"]