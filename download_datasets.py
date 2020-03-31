from torchvision.datasets import CIFAR10, CIFAR100
import requests
import os
import zipfile

DATA_DIR = "/home_local/saareliad/data"

# DEFAULT_DATA_DIR = os.path.expanduser('~/.pytorch-datasets')
# DATA_DIR = DEFAULT_DATA_DIR


def download_file(url, DATA_DIR=DATA_DIR):
    local_filename = url.split('/')[-1]
    local_filename = os.path.join(DATA_DIR, local_filename)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


def download_wiki2():
    URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    path_to_zip_file = download_file(URL)
    print(f"-I- Donwloaded wikitext2 to {path_to_zip_file}. Extracting...")
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    # NOTE: The data will be in DATA_DIR/wikitext-2-raw
    print("-I- Done")


def download_squad():
    """ Download Squad datasets """

    script = f"""\
pushd {DATA_DIR}
if [ ! -d squad1 ] ; then 
mkdir squad1
cd squad1
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
cd ..

fi

if [ ! -d squad2 ] ; then 
mkdir squad2
cd squad2
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
curl https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ > evaluate-v2.0.py
cd ..
fi
popd
"""
    os.system("bash -c '%s'" % script)


if __name__ == "__main__":
    CIFAR100(root=DATA_DIR, download=True, train=True)
    CIFAR100(root=DATA_DIR, download=True, train=False)

    CIFAR10(root=DATA_DIR, download=True, train=True)
    CIFAR10(root=DATA_DIR, download=True, train=False)

    download_wiki2()
    download_squad()
