from pipeline.hadmid_reader import Hadm_Id_Reader

print("beginning to read all files in")
reader = Hadm_Id_Reader("./data/rawdatafiles/byHadmID/")

# reader.countEventsPerHospitalAdmission().to_csv("data/rawdatafiles/countsByHadmid.csv")
print(reader.resample_fixed_length(hadmid=str(100001)))
