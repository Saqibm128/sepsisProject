#@author Ying Sha

import psycopg2
import pandas as pd

def get_patient_list(if_server):
    """
    :param if_server: whether run on your laptop or on the lab servers
    :return: detailed admission infor for selected patients having multiple hospital visits
    """
    try:
        if if_server:
            conn = psycopg2.connect("dbname='mimic' user='mimic3db' host='qdqc.bme.gatech.edu' password='m1m1c3db' port='5432' " )
        else:
            conn = psycopg2.connect("dbname='mimic' user='mimic3db' host='localhost' password='m1m1c3db' port='2000' ")
    except:
    	raise
    cur = conn.cursor()
    cur.execute("SET search_path TO mimiciii")

    #Select patients having multiple hospital visits
    query = 'SELECT admissions.subject_id, COUNT(admissions.subject_id) ' \
        + 'FROM admissions ' \
        + 'GROUP BY admissions.subject_id '\
        + 'HAVING COUNT(admissions.subject_id) > 1'

    #Get subject ID of those patients
    patient_multi_admit = pd.read_sql(query, conn)
    #Create queries for admission details of those patients
    temp = ', '.join(str(i) for i in patient_multi_admit.subject_id)
    query = 'SELECT * FROM admissions ' \
           + 'WHERE admissions.subject_id IN ( ' + temp + ' ) '

    details = pd.read_sql(query, conn)

    return details

print(get_patient_list(if_server=False))
