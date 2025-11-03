import os
import pandas as pd
import kagglehub

class DataDownloader:
    def __init__(self, dataset_name="new-york-city/ny-311-service-requests", data_dir="data"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_dataset(self):
        print("Downloading or loading dataset from cache...")
        path = kagglehub.dataset_download(self.dataset_name)
        print(f"Dataset available at: {path}")
        return path

    def load_dataframe(self):
        dataset_path = self.download_dataset()
        csv_file = None
        for file in os.listdir(dataset_path):
            if file.endswith(".csv"):
                csv_file = os.path.join(dataset_path, file)
                break
        if not csv_file:
            raise FileNotFoundError("No CSV file found in dataset folder!")

        df = pd.read_csv(csv_file)
        print(f"Data loaded with shape: {df.shape}")

        # Save a copy in your project folder
        local_copy = os.path.join(self.data_dir, "raw_311_data.csv")
        df.to_csv(local_copy, index=False)
        print(f"Local copy saved at: {local_copy}")
        return df
