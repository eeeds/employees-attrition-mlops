FROM  python:3.9.12
LABEL Author, Esteban Encina
RUN mkdir app models
COPY requirements.txt ./requirements.txt
## Install dependencies
RUN pip install -r requirements.txt

## Run the app
CMD streamlit run app/streamlit_app.py
