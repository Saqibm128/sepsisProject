import pickle
import commonDB
import pandas as pd
# import waveformUtil as wfutil TODO: remove these circular dependencies
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for getting out all results from sql queries.
##      It does not do much else, because getting the queries
##      takes a lot of time. Once done, it places the data as a raw pickle file

def getQSofaCategorization(hadm_ids = None, subject_ids = None):
    """
    This function uses the quick sepsis organ failure assessment, as defined by the third international
    consensus for sepsis
    :param hadm_ids a list of hadm_ids which to apply standard to
    :param subject_ids a list of subject_ids which to apply standard to. if hadm_ids is set, this doesn't do anything
    :return dataframe with hadm_ids as index and a column with sepsis
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4968574/
    """

    if hadm_ids is None and subject_ids is None:
        hadm_ids = commonDB.read_sql("SELECT hadm_id FROM admissions")
    elif hadm_ids is None:
        hadm_ids = commonDB.read_sql("SELECT hadm_id FROM admissions WHERE subject_id in " \
                                + commonDB.convertListToSQL(subject_ids), conn)
    results = pd.DataFrame()
    #check and see if glasgow coma score is below 15
    alteredMentalVerbal = commonDB.read_sql("SELECT hadm_id FROM chartevents " \
                                            + "WHERE valuenum < 15 and itemid = 227013 and hadm_id in " \
                                            + commonDB.convertListToSQL(hadm_ids))
    increasedRespiration = commonDB.read_sql("SELECT hadm_id FROM chartevents " \
                                            + "WHERE valuenum >= 22 and itemid = 224690 and hadm_id in" \
                                            + commonDB.convertListToSQL(hadm_ids))
    increasedSystolicBP = commonDB.read_sql("SELECT hadm_id FROM chartevents " \
                                            + 'WHERE valuenum <= 100 and itemid in (228152, 220050, 442, 455, 6, 51, 3313, 3317, 3319, 3321, 3323) and hadm_id in ' \
                                            + commonDB.convertListToSQL(hadm_ids))
    #TODO: Complete this later
    return None
def getCategorizations(hadm_ids = None):
    """
    This is a function to run the Angus.sql query, which is responsible for categorizing patients
    :postcondition stores results as data/rawdatafiles/classifiedAngusSepsis.p
    :param ids the hadm_ids to look at when doing the query; if None, run it on all
    :return the angusData
    """
    conn = commonDB.getConnection()
    with open("data/sql/angus.sql") as f:
        query = f.read()
    if hadm_ids is None:
        query = query.replace("<INSERT IDS HERE>", "")
    else:
        query = query.replace("<INSERT IDS HERE>", "WHERE hadm_id IN " + commonDB.convertListToSQL(hadm_ids))

    angusData = pd.read_sql(query, conn)
    angusData.set_index(["hadm_id"], inplace=True)
    return angusData
if __name__ == "__main__":
    getCategorizations(True)
