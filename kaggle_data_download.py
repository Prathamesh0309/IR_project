import os
import pandas as pd

class DataDownloader:
    '''
    Class to handle loading of the dataset from a local CSV file.'''

    def __init__(self, local_path="data/311-service-requests-from-2010-to-present.csv"):
        self.local_path = local_path

    def load_dataframe(self):
        '''
        Load the dataset from the local CSV file into a pandas DataFrame.
        '''
        # Check if the file exists
        if not os.path.exists(self.local_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {self.local_path}. "
                f"Please place your manually downloaded CSV there."
            )
        # Load only relevant columns to save memory
        cols = [
        "Unique Key",
        "Created Date",
        "Complaint Type",
        "Descriptor",
        "Agency",
        "Agency Name",
        "Borough",
        "Incident Zip"
    ]
        # Load the dataset
        df = pd.read_csv(self.local_path, usecols=cols, low_memory=True)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

        return df
