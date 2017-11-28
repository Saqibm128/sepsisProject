from readWaveform.waveform_reader import Waveform_Reader

reader = Waveform_Reader("/data/mimic3wdb/matched")
print(reader.access_subject("p010013"))
