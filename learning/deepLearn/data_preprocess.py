import wfdb
import csv
import numpy as np
import seaborn as sns
import pickle
sns.set(color_codes=True)


def pad_and_truncate(input, desired_length=9000, pad_pos='before', pad_value=0):
    if len(input) >= desired_length:
        return input[0:desired_length]
    elif pad_pos == 'before':
        return np.pad(input, (desired_length - len(input), 0), 'constant',constant_values=pad_value)
    elif pad_pos == 'after':
        return np.pad(input, (0, desired_length - len(input)), 'constant', constant_values=pad_value)


if __name__ == '__main__':

    # Read in all patient IDs and corresponding labels
    with open('../training2017/REFERENCE.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        patient_ID, labels = [],[]
        for row in reader:
            name = row['ID']
            lbl = row['Label']
            if (name is None or name =='') or (lbl is None or lbl ==''):
                continue
            patient_ID.append(name)
            labels.append(lbl)
    f.close()
    patient_ID = np.array(patient_ID)
    labels = np.array(labels)
    label_order, int_labels = np.unique(labels, return_inverse=True) # The order is A, N, O, ~

    #TODO: consider creating a mask
    #TODO: pack_packed_sequence
    # Check the distribution of the length of all ECG samples
    # max length is 18286, the majority of them are 9000
    max_length, length_list, raw_data = 0, [], []
    for i in range(len(patient_ID)):
        signals, fields = wfdb.srdsamp('../training2017/'+patient_ID[i])
        length_list.append(len(signals))
        signals = np.squeeze(np.array(signals, dtype='f'))
        raw_data.append(signals)
    length_list = np.array(length_list)
    sns.distplot(length_list)
    print(max_length)
    pickle.dump(raw_data,open('../data/raw_data.pkl', 'wb'))
    # data = np.asarray([pad_and_truncate(a, desired_length=length_list[order_index[0]], pad_pos='after', pad_value=0) for a in raw_ordered_data])
    # new_int_labels = [int_labels[i] for i in order_index]
    #with open('../data/new_labels.pkl', 'wb') as f:
    #    pickle.dump(new_int_labels, file=f)
    # Pad raw data with 0, desired input dim, truncate
    data = np.asarray([pad_and_truncate(a, desired_length=max_length, pad_pos='after', pad_value=0) for a in raw_data])
    with open('../data/labels.pkl', 'wb') as f:
        pickle.dump(int_labels, file=f)
    with open('../data/new_padded_data.pkl', 'wb') as f:
        pickle.dump(data, file=f)

