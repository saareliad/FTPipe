import os
import zipfile

import requests


# DEFAULT_DATA_DIR = os.path.expanduser('~/.pytorch-datasets')
# DATA_DIR = DEFAULT_DATA_DIR


def download_file(url, DATA_DIR=""):
    local_filename = url.split('/')[-1]
    local_filename = os.path.join(DATA_DIR, local_filename)
    if os.path.exists(local_filename):
        print(f"-I- file {local_filename} already exists, skipping download.")
        return local_filename
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


def download_wiki2(DATA_DIR=""):
    URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    path_to_zip_file = download_file(URL)
    print(f"-I- Donwloaded wikitext2 to {path_to_zip_file}. Extracting...")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # NOTE: The data will be in DATA_DIR/wikitext-2-raw
    print("-I- Done")


def download_wiki103(DATA_DIR=""):
    # FIXME: raw or not? in my other dir I used without the raw.
    # "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    # raw dir has tokens
    URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

    path_to_zip_file = download_file(URL)
    print(f"-I- Donwloaded wikitext103 to {path_to_zip_file}. Extracting...")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # NOTE: The data will be in DATA_DIR/wikitext-103-raw
    print("-I- Done")


if __name__ == "__main__":
    download_wiki2()
    # download_wiki103() TODO: update this when we have experiment.
