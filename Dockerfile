FROM python:3.9.12
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "requirements.txt", "./" ]

RUN pip install -r requirements.txt

COPY ["model.py", "./" ]

CMD python model.py
