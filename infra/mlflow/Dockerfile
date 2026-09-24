# Serveur MLflow : suivi des expériences, registre des modèles et stockage des artefacts.
# Image construite ici pour ajouter l'authentification, le pilote PostgreSQL et le client S3.
FROM python:3.12.14-slim

RUN pip install --no-cache-dir \
    "mlflow[auth]==3.16.1" \
    "psycopg2-binary==2.9.12" \
    "boto3==1.42.30" \
 && useradd --create-home --uid 10001 mlflow
USER mlflow
WORKDIR /home/mlflow
EXPOSE 5000
