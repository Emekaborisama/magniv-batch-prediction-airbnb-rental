import pandas as pd
import os
s3_url = "s3://mangivprojectdata/data_file/data.csv"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

def upload_s3(data):
    """ upload data to s3"""
    upload_data = data.to_csv(s3_url, index=False, storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY
    })
    print(upload_data)
    return upload_data
    
    
def download_s3():
    """download data from s3"""
    download_data = pd.read_csv(s3_url, storage_options={
        "key": AWS_ACCESS_KEY_ID,
        "secret": AWS_SECRET_ACCESS_KEY
    })
    return download_data