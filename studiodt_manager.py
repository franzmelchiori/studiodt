"""
    StudioDT Manager
    Francesco Melchiori, 2025
"""


import os
import pandas


PATH_TO_START = os.getcwd()
SUFFIX_TO_FIND = '_medical_record'


class PatientMedicalRecordPaths:
    """ PatientFolders gets all the folders within the given path and checks 
        each of them in order to filter in legit medical record of patients 
        based on their filename, lastly providing the list of their paths
    """
    def __init__(self, path_to_start=PATH_TO_START, suffix_to_find=SUFFIX_TO_FIND):
        self.path_to_start = path_to_start
        self.suffix_to_find = suffix_to_find
        self.medical_records = self.find_medical_records()

    def __repr__(self):
        message = ''
        for medical_record in self.medical_records:
            message += medical_record
            message += '\r\n'
        return message

    def get_folders(self):
        """ get all the folders within the given path
        """
        folders = []
        with os.scandir(self.path_to_start) as path_objects:
            for path_object in path_objects:
                if not path_object.name.startswith('.') and path_object.is_dir():
                    folders.append(path_object.name)
        return folders

    def find_medical_records(self):
        """ find .xlsx files with the given suffix within the given paths
        """
        folders = self.get_folders()
        medical_records = []
        for folder in folders:
            with os.scandir(self.path_to_start + '\\' + folder) as path_objects:
                for path_object in path_objects:
                    if path_object.is_file():
                        filename = path_object.name.split('.')[0]
                        if filename.endswith(self.suffix_to_find):
                            medical_records.append(folder + '\\' + path_object.name)
        return medical_records


if __name__ == '__main__':
    pmrp = PatientMedicalRecordPaths()
    print(pmrp)
