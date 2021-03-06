import wget
import tarfile
import os

def download_file(url):
    if not url.split('/')[-1] in os.listdir(os.getcwd()):
        filename = wget.download(url)
        print(f'\n{filename} - Downloaded!')
    else:
        filename = url.split('/')[-1]
        print(f"{filename} - Already exists !")
    return filename

def extract_file(filename):
    # if not url.split('/')[-1] in os.listdir(os.getcwd()):
    if filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
    elif filename.endswith("tar"):
        tar = tarfile.open(filename, "r:")
        tar.extractall()
        tar.close()

    print(f'{filename} - Extracted!')
    

# 
if __name__ == "__main__":
    urls = [
        'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'https://www.openslr.org/resources/12/train-clean-360.tar.gz',
        'https://www.openslr.org/resources/12/dev-clean.tar.gz', 
        'https://www.openslr.org/resources/12/test-clean.tar.gz'
    ]

    for url in urls:
        print(url)
        filename = download_file(url)
        extract_file(filename)
        print('\n')
