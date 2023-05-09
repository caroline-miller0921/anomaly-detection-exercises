import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
import env
from sklearn import metrics
import os

def get_logs():

    url = env.get_db_url('curriculum_logs')
    if os.path.exists('logs.csv'):
        logs = pd.read_csv('logs.csv', index_col = 0)
    else:
        logs = pd.read_sql('select * from logs;', url)
        logs.to_csv('logs.csv')
    return logs
    
def get_cohorts():

    url = env.get_db_url('curriculum_logs')
    if os.path.exists('cohorts.csv'):
        cohorts = pd.read_csv('cohorts.csv', index_col = 0)
    else:
        cohorts = pd.read_sql('select * from cohorts;', url)
        cohorts.to_csv('cohorts.csv')
    return cohorts

def merge_logs_and_cohorts():

    logs = get_logs()
    cohorts = get_cohorts()
    df = logs.merge(cohorts, left_on='cohort_id', right_on='id')
    return df

def prepare_df():

    df = merge_logs_and_cohorts()
    df = df.drop(columns={'id', 'created_at', 'updated_at', 'deleted_at', 'program_id', 'slack'})
    df['path'] = df['path'].str.split(pat='/')
    df['module'] = df['path'].str[0]
    df['lesson'] = df['path'].str[-1]
    df['timestamp'] = df['date'] +' ' + df['time']
    df['date'] = pd.to_datetime(df['date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['module'] != '']
    df['module'] = df['module'].astype(str)
    df['web_dev'] = np.where(df.module.str.contains('java|jquery|html|jquery|html|css|web', regex=True) ==True, 1, 0)
    df['data_science'] = np.where(df.module.str.contains('sql|python|git|classification|regression|anomaly|clustering|stats|storytelling|timeseries|distributed-ml|Regression|scientist|pandas|Stats|AI-ML-DL|Cluster|Excel|control_structures|functions|ordinary_least_squares|Modeling|Probability|matplotlib|Explore|imports|Correlation', regex=True) ==True, 1, 0)
    df = df.set_index('timestamp').sort_index()
    df = df.dropna()
    return df

def get_fences(df, col, k=1.5):
    '''
    get_fences takes in a dataframe and a string literal
    df, a pandas dataframe
    k, an integer representing the fence for our method
    col, a string literal represening a column name
    returns the lower and upper fences of a series
    '''
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    lower_fence = q1 - iqr*k
    upper_fence = q3 + iqr*k
    return lower_fence, upper_fence
