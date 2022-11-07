
# Import PySpark Pandas


from distutils.command.upload import upload
import os
import resource
import requests
import pandas as pd
from magniv.core import task
# from pyspark.sql import SparkSession
# import pyspark.pandas as ps
import pickle
# import pyspark.pandas as ps
# from pyspark.context import SparkContext
# sc = SparkContext.getOrCreate()
# spark = SparkSession.builder.appName("SparkByExamples.com").getOrCreate()
#read pickled model via pipeline api
from upload_download_s3 import download_s3, upload_s3
serialized_model = open("tasks/model/model_lin.p", "rb")
model = pickle.load(serialized_model)
url = os.getenv("DATA_URL")
# model_bd = sc.broadcast(model)





def inference(data,model):
    """ load model, preprocess data and run price prediction inference"""
    data['fields.last_review'] = pd.to_datetime(data['fields.last_review'])
    data['fields.last_review_month'] =data['fields.last_review'].dt.month
    data['fields.last_review_days'] = data['fields.last_review'].dt.day
    data['fields.last_review_year'] = 2022 - data['fields.last_review'].dt.year
    data= data.drop(['fields.last_review','fields.latitude', 'fields.longitude'],axis=1)
    data['fields.room_type'] = data['fields.room_type'].replace({"Entire home/apt":0,"Private room":1,
                                                   "Shared room":2,"Hotel room":3})
    return model.predict(data.values)
   


@task(key='first', schedule="@monthly",on_success=["second"], description=" get airbnb data and store it on s3")
def get_data():
    payload={'dataset': 'airbnb-listings',
    'q': 'new york'}
    response = requests.request("POST", url,data=payload, )
    result = response.json()
    result_ori = pd.json_normalize(result['records'])
    res = result_ori[['fields.room_type', 'fields.minimum_nights', 'fields.number_of_reviews', 'fields.reviews_per_month',
       'fields.host_listings_count', 'fields.availability_365','fields.latitude', 'fields.longitude',
       'fields.last_review']]
    up_data = upload_s3(res)
    return res, result_ori



@task(key="second",schedule="@monthly",resources={"cpu": "2000m", "memory": "2Gi"},description=" preprocess data and run price prediction inference")
def merge_result_geo():
    data = download_s3()
    pred = inference(model=model, data=data)
    data['predicted_prices'] = pred
    return data[['predicted_prices','fields.latitude', 'fields.longitude' ]].to_json()


if __name__ == '__main__':
    merge_result_geo()
