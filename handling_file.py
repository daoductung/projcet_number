import os
import soundfile
import wavy
import shutil
import csv


# Todo: get path folder
def get_path_folder():
    return os.getcwd()


path_root = get_path_folder()
path_data = get_path_folder() + '\\data\\'
path_train = get_path_folder() + '\\train\\'
path_record = get_path_folder() + '\\record\\'


# TODO: rename file
def rename_file():
    for i in range(0, 12):
        for j in range(0, 10):
            file_name = path_data + 'data_' + str(i) + '\\' + str(
                j) + '.wav'
            new_file_name = path_data + 'data_' + str(i) + '\\' + str(
                j) + str(
                i) + '.wav'
            os.rename(file_name, new_file_name)


# Todo: create folder
def create_folder():
    for i in range(0, 10):
        os.makedirs(str(i))


# Todo: copy file

def coypy_file():
    for i in range(0, 12):
        for j in range(0, 10):
            original = path_data + 'data_' + str(i) + '\\' + str(j) + str(
                i) + '.wav'
            target = path_train + str(j) + '\\' + str(j) + str(
                i) + '.wav'
            shutil.copyfile(original, target)


# Todo: Convert .wav -> 16bit

def convert_to_16b():
    for i in range(0, 10):
        for j in range(0, 12):
            data, samplerate = soundfile.read(path_train + str(i) + '\\' + str(i) + str(j) + '.wav')
            soundfile.write(path_train + str(i) + '\\' + str(i) + str(j) + '.wav', data, samplerate,
                            subtype='PCM_16')


# Todo: check file 16bit


# print(wavy.info(path + 'train/0/converted_00.wav'))

# print(get_path_folder())

# Todo: create csv

def create_file_csv(file_name):
    with open(file_name, mode='w') as csv_file:
        fieldnames = ['file_name', 'classID', 'className']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0, 10):
            for j in range(12):
                writer.writerow(
                    {'file_name':  str(i) + '\\' + str(i) + str(j) + '.wav', 'classID': i, 'className': i})
        csv_file.close()

create_file_csv('title.csv')
