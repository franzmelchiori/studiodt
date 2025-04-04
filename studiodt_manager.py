"""
    StudioDT Manager
    Francesco Melchiori, 2025
"""


import os
import pandas


class PatientFolders:
    """ PatientFolders gets all the folders within the given path and checks 
        each of them in order to filter in legit medical record of patients 
        based on their filename, lastly providing the list of their paths
    """
    def __init__(self, path_to_start='.', suffix_to_find='bla'):
        if path_to_start == '.':
            path_to_start = os.getcwd()
        else:
            path_to_start = 'bla'
        print(path_to_start)

    def get_folders():
        """ get all the folders within the given path
        """
        pass

    def find_medical_record():
        """ find a .xlsx file with the given suffix within the given path
        """
        pass


if __name__ == '__main__':
    pf = PatientFolders()
