# VERSION 2.0
# AUTHOR: Patrick Alves
# DESCRIPTION: Basic Airflow 2.0 container
# BUILD: docker build --rm -t cpatrickalves/python37-airflow2 .
# SOURCE: https://github.com/cpatrickalves/airflow2-spam-classifier-pipeline
FROM python:3.7-slim

ENV AIRFLOW_HOME=/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
WORKDIR ${AIRFLOW_HOME}

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN airflow db init && \
    airflow users create \
        --username admin \
        --password admin \
        --firstname Patrick \
        --lastname Alves \
        --role Admin \
        --email cpatrickalves@gmail.com

#ARG AIRFLOW_USER_HOME=/usr/local/airflow
#ENV AIRFLOW_HOME=${AIRFLOW_USER_HOME}

COPY scripts/entrypoint.sh ./entrypoint.sh
COPY dags/ ./dags

#COPY config/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg
EXPOSE 8080 5555 8793

CMD ["./scripts/entrypoint.sh"]