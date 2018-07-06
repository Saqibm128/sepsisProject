import pickle
import commonDB
import pandas as pd
# import waveformUtil as wfutil TODO: remove these circular dependencies
# from matplotlib import pyplot as plt


## @author Mohammed Saqib
## This file is responsible for categorizing sepsis as per angus criterion
def getQSofaCategorization(hadm_ids = None, subject_ids = None):
    """
    This function uses the quick sepsis organ failure assessment, as defined by the third international
    consensus for sepsis
    TODO: Complete this later
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
def getCategorizationsBySubjectID(catDF = None):
    """
    This is a function to return the categorizations by subject instead of hadm_id
    in other words, if a subject had a sepsis diagnosis ever, he/she will be classfied as such
    :param optional categorization DataFrame to use that uses hadm_id based categorization
    :return the angus categorization
    """
    if catDF == None:
        conn = commonDB.getConnection()
        with open("data/sql/angus.sql") as f:
            query = f.read()
            query = query.replace("<INSERT IDS HERE>", "")
        catDF = pd.read_sql(query, conn)
        catDF.set_index(["hadm_id"], inplace=True)
    bySubjectCat = []
    for subject_id in catDF["subject_id"].unique():
        onlySubject = catDF[catDF["subject_id"] == subject_id]
        if onlySubject.apply(lambda row: row["angus"] == 1, axis = 1).any():
            bySubjectCat.append(pd.DataFrame({"subject_id": [subject_id], "angus": [1]}))
        else:
            bySubjectCat.append(pd.DataFrame({"subject_id": [subject_id], "angus": [0]}))
    bySubject = pd.concat(bySubjectCat)
    return bySubject.set_index("subject_id")


def getCategorizations(hadm_ids = None):
    """
    This is a function to run the Angus.sql query, which is responsible for categorizing patients
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

if __name__=="__main__":
    getCategorizations().to_csv("data/rawdatafiles/categorizations.csv")
