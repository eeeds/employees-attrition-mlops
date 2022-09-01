FROM  python:3.9.12
LABEL Author, Esteban Encina

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY evidently_app.py .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8501"]
