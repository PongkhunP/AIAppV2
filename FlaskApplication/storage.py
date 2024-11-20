from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
import boto3
import os
import mariadb

load_dotenv()
host=os.getenv('DB_HOST')
port=int(os.getenv('DB_PORT'))
user=os.getenv('MARIADB_USER')
password=str(os.getenv('MARIADB_PASSWORD'))
database=os.getenv('DB_DATABASE')

bucket_url=os.getenv('BUCKET_URL')

client = boto3.client(
    service_name ="s3",
    endpoint_url = os.getenv('END_POINT_URL'),
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('BUCKET_REGION'),
)

def get_db_connection():
    try:
        conn = mariadb.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to the database: {e}")
        return None
