import requests
import subprocess
import tarfile
from tqdm import tqdm
import os


def download_file(url, local_filename=None):
    if local_filename is None:
        # The default is to save the file to the current directory
        local_filename = url.split('/')[-1]
    local_filename = os.path.expanduser(local_filename)
    print(f"Downloading {url} to {local_filename}")
    # Get the directory path and check if it exists
    directory = os.path.dirname(local_filename)
    if directory and not os.path.exists(directory):
        print(f"Creating {directory}")
        os.makedirs(directory, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(local_filename, 'wb') as f, tqdm(
        desc=local_filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                progress_bar.update(len(chunk))

    return local_filename

def extract_tar(filename):
    filename = os.path.expanduser(filename)
    directory = os.path.dirname(filename)
    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall(path=directory)
    print(f"Extraction complete.")

def clone_repository(repo_url, target_directory):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory, exist_ok=True)
        print(f"Created directory: {target_directory}")
    try:
        # Run the git clone command
        # We use shell=True for simplicity, but in a production app,
        # passing a list is often safer.
        print(f"Cloning {repo_url} into {target_directory}...")
        subprocess.run(
            ["git", "clone", repo_url, target_directory],
            check=True,
            capture_output=True,
            text=True
        )
        print("Clone successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during clone: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Git is not installed on this system.")
        return False
        # Example usage:
        # clone_repository("https://github.com/user/repo.git", "./models/my_repo")
