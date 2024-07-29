import opendatasets as od


class DatasetDownloader:
    """
    DatasetDownloader is a class to handle downloading datasets from various sources,
    including Kaggle, using the opendatasets library.

    Attributes:
        download_url (str): The URL of the dataset to download.

    Methods:
        download_dataset(): Downloads the dataset from the specified URL.

    Example usage:
        downloader = DatasetDownloader("https://www.kaggle.com/datasets/tunguz/us-elections-dataset")
        downloader.download_dataset()
    """
    def __init__(self, dataset_url):
        """
        Initializes the DatasetDownloader with the dataset URL.

        Parameters:
            dataset_url (str): The URL of the dataset to download.
        """
        self.download_url = dataset_url

    def download_dataset(self):
        """
        Downloads the dataset from the specified URL using the opendatasets library.
        """
        od.download(self.download_url)
        print(f"Dataset downloaded successfully")
