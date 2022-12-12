FROM python:3.9 
WORKDIR /Deployement_LMR_App
COPY ./requirements.txt /Deployement_LMR_App/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /Deployement_LMR_App/requirements.txt
COPY ./app /Deployement_LMR_App/app
COPY ./Customs_Codes_For_ML /Deployement_LMR_App/Customs_Codes_For_ML
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]


