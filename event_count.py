from pipeline.hadmid_reader import Hadm_Id_Reader

print("beginning to read all files in")
reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID3/")

# reader.countEventsPerHospitalAdmission().to_csv("data/rawdatafiles/countsByHadmid.csv")
reader.traditional_time_event_matrix().to_csv("data/rawdatafiles/full_data_matrix.csv")
