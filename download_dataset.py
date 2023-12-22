from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import requests
import json
import pandas as pd
from tqdm import tqdm
import os

# Parameters
cvat_url = 'http://localhost:8080'  # Replace with your CVAT URL
username = 'username'  # Replace with your CVAT username
password = 'password'  # Replace with your CVAT password
export_format = 'YOLO 1.1'  # Replace with your desired export format
remap_dict = {  # Replace with your desired class IDs mapping
    'person': '0',
    'car': '1',
    'truck': '2',
    'loader': '2',
    'dozer': '2',
    'excavator' : '2',
    'sideboom': '2',
    'forklift': '2',
    'motorgrader': '2',
    'pipe': '3'
    }

# Authenticate
def authenticate(cvat_url, username, password):
    """
    Authenticate to CVAT.

    Args:
    cvat_url (str): The URL of the CVAT server.
    username (str): The username of the CVAT user.
    password (str): The password of the CVAT user.

    Returns:
    requests.Session: The authenticated session.
    """
    session = requests.Session()
    headers = {'Content-Type': 'application/json'}
    auth_response = session.post(f"{cvat_url}/api/auth/login", 
                                 data=json.dumps({'username': username, 'password': password}),
                                 headers=headers)
    auth_response.raise_for_status()
    return session


session = authenticate(cvat_url, username, password)

def fetch_all_pages(url):
    """Fetch all pages from a paginated API."""
    results = []
    while url:
        response = session.get(url)
        response.raise_for_status()
        data = response.json()
        results.extend(data['results'])
        url = data.get('next')
    return results


def fetch_page(url_with_page):
    """Fetch a single page from a paginated API."""
    response = session.get(url_with_page)
    response.raise_for_status()
    return response.json()


def fetch_specific_page(url, page_number):
    """Fetch a specific page from a paginated API."""
    response = session.get(url, params={'page': page_number})
    response.raise_for_status()
    return response.json()


# list Projects
projects = fetch_all_pages(f"{cvat_url}/api/projects")
projects_df = pd.DataFrame(projects)

# get project names
project_names = {project['id']: project['name'] for project in projects}


# list Tasks
tasks = fetch_all_pages(f"{cvat_url}/api/tasks")
tasks_df = pd.DataFrame(tasks)

tasks_df['project_name'] = tasks_df['project_id'].map(projects_df.set_index('id')['name'])



# Now you can process the dataframes further as needed
print(projects_df.head())  # Display first few rows of the projects dataframe
print(tasks_df.head())    # Display first few rows of the tasks dataframe

# You can save these dataframes to a file, for example, a CSV file
projects_df.to_csv('cvat_projects.csv', index=False)
tasks_df.to_csv('cvat_tasks.csv', index=False)

def get_task_info(task_id):
    """
    Get information about a task.

    Args:
    task_id (int): The ID of the task to get information for.

    Returns:
    dict: The task information.
    """
    response = session.get(f"{cvat_url}/api/tasks/{task_id}")
    response.raise_for_status()
    return response.json()

def request_download_annotation(task_id, file_format='YOLO 1.1', save_images=False):
    """
    Request server to prepare the annotation dump.

    Args:
    task_id (int): The ID of the task to request annotations for.
    file_format (str): The format of the annotation file (default: 'YOLO 1.1').
    save_images (bool): Whether to also save the images (default: False).

    Returns:
    bool: success status.
    """
    endpoint = 'dataset' if save_images else 'annotations'

    dump_url = f"{cvat_url}/api/tasks/{task_id}/{endpoint}?format={file_format}"
    response = session.get(dump_url)
    
    if response.status_code == 202:
        print(f"Dump of annotations from task {task_id} has been started. Awaiting completion...")
        return False
    elif response.status_code == 201:
        print(f"Annotations file from task {task_id} is ready to download.")
        return True
    else:
        response.raise_for_status()


def download_annotation(task_id, file_format, save_path, save_images=False):
    """
    Download annotations for a given task ID from CVAT.

    Args:
    task_id (int): The ID of the task to download annotations for.
    file_format (str): The format of the annotation file (e.g., 'CVAT XML 1.1 for images').
    save_path (str): The path where the annotation file will be saved.
    save_images (bool): Whether to also save the images (default: False).
    """
    while not request_download_annotation(task_id, file_format, save_images=save_images):
        time.sleep(1)

    endpoint = 'dataset' if save_images else 'annotations'

    # The URL to download the annotation
    annotation_url = f"{cvat_url}/api/tasks/{task_id}/{endpoint}?format={file_format}&action=download"
    headers = {'Content-Type': 'application/json'}

    # Initial request to get the content-length header for total size
    with session.get(annotation_url, headers=headers, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192  # size of each chunk

        # Setup the progress bar
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading annotations for task {task_id}")

        # Download with progress bar
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("[ERROR] Transferred file is corrupted")
        exit(1)

    print(f"Annotations for task {task_id} downloaded successfully.")


def save_images_from_disk_video(video_path, output_path):
    """
    Get images for from a video on the disk.
    Args:
    video_path (str): The path of the video file.
    output_path (str): The path where the images will be saved.
    """
    if not os.path.exists(video_path):
        print(f'[ERROR] {video_path} does not exist.')
        raise FileNotFoundError
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    images_format = os.path.join(output_path, 'frame_%06d.png')
    subprocess.run(['ffmpeg', '-i', video_path, '-start_number', '0', '-b:v', '10000k', '-vsync', '0', '-an', '-y', '-q:v', '16', images_format])


def get_video_for_task_from_disk(task_name, videos_path):
    """
    Get the video for a task from the disk.
    Args:
    task_name (str): The name of the task.
    videos_path (str): The path where the videos are saved.
    """
    videos_path = Path(videos_path)
    video_path = list(videos_path.rglob(f'{task_name}.avi'))
    if len(video_path) == 0:
        print(f'[ERROR] {task_name}.avi does not exist.')
        raise FileNotFoundError
    
    return video_path[0]


def unzip_annotation(zip_path, extract_path, remove_zip=True):
    """
    Extract the contents of a ZIP file.

    Args:
    zip_path (str): The path of the ZIP file to be extracted.
    extract_path (str): The path where the contents of the ZIP file will be extracted.
    remove_zip (bool): Whether to remove the ZIP file after extraction (default: True).
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    if remove_zip:
        os.remove(zip_path)

    print(f"Annotations extracted to {extract_path}")

def remap_annotation(raw_dataset_path):
    """
    Remap the annotation files to match the desired class IDs.
    Warning: This function will overwrite the original annotation files.

    Args:
    raw_dataset_path (str): The path of the extracted annotation files.
    """
    raw_dataset_path = Path(raw_dataset_path)
    # Get raw class IDs
    raw_class_ids = {}
    with open(raw_dataset_path / "obj.names") as file:
        for index, line in enumerate(file):
            raw_class_ids[index] = line.strip()

    # Skip remap if the class IDs are identical
    skip_remap = True
    for index, class_id in raw_class_ids.items():
        class_id = class_id.lower()
        if remap_dict.get(class_id) is None:
            print(f"[WARN] Class {class_id} not found in remap dictionary. Raw class ID {index} will be used and may cause errors.")
            skip_remap = False
            break
        if index != int(remap_dict[class_id]):
            skip_remap = False
            break
    
    # Get the list of annotation files
    if not skip_remap:
        annotation_files = list((raw_dataset_path / "obj_train_data").glob('*.txt'))
        for filename in annotation_files:
            with open(filename) as file:
                lines = file.readlines()

            with open(filename, "w") as file:
                for line in lines:
                    class_id, *coords = line.split()
                    # use raw id if class not found in remap dictionary
                    class_id = remap_dict.get(raw_class_ids[int(class_id)], class_id)
                    file.write(f"{class_id} {' '.join(coords)}\n")

        print(f"Remapped annotations.")

def prepare_dataset(raw_dataset_path, metadata, output_path=None, split=None, offline_videos_path=None):
    """
    prepare the dataset and metadata.
    If output_path is not specified, the dataset will be prepared in-place and cache will be removed.
    If split is not specified, the dataset will be prepared without a subdirectory.
    Args:
    raw_dataset_path (str): The path of the extracted annotation files.
    metadata (dict): The metadata of the dataset task.
    output_path (str): The path where the dataset will be prepared.
    split (str): The split of the dataset (e.g., 'train', 'val', 'test').
    offline_videos_path (str): The path where the offline videos are saved.
    """
    raw_dataset_path = Path(raw_dataset_path)
    inplace = output_path is None
    output_path = Path(raw_dataset_path).parent / metadata["project_name"] if inplace else Path(output_path)
    
    # Create the output directory
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path/".meta").mkdir(parents=True, exist_ok=True)
    with open(f'{output_path}/.meta/{metadata["id"]}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Copy the annotation files
    annotation_path = output_path / "labels" if split is None else output_path / split / "labels"
    annotation_path.mkdir(parents=True, exist_ok=True)
    annotation_files = list((raw_dataset_path / "obj_train_data").glob('*.txt'))

    if len(list(annotation_files)) == 0:
        print("[WARN] No annotations found in the dataset.")

    for filename in annotation_files:
        new_filename = f"{metadata['project_id']}_{metadata['id']}_{filename.name}"
        shutil.copy(filename, annotation_path / new_filename)

    # Copy the image files
    image_path = output_path / "images" if split is None else output_path / split / "images"
    image_path.mkdir(parents=True, exist_ok=True)
    IMG_FORMATS = '.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'
    image_files = list(p.resolve() for p in Path(raw_dataset_path / "obj_train_data").glob("**/*") if p.suffix.lower() in IMG_FORMATS)

    if len(image_files) == 0:
        # Get images from disk if available
        if offline_videos_path is not None:
            print(f"Extracting images from {video_path}")
            video_path = get_video_for_task_from_disk(metadata['name'], offline_videos_path)
            save_images_from_disk_video(video_path, Path(raw_dataset_path / "obj_train_data"))

            image_files = (p.resolve() for p in Path(raw_dataset_path / "obj_train_data").glob("**/*") if p.suffix.lower() in IMG_FORMATS)
            if len(list(image_files)) == 0:
                print("[WARN] No images found in the dataset.")
        else:
            print("[WARN] No images found in the dataset.")

    for filename in image_files:
        new_filename = f"{metadata['project_id']}_{metadata['id']}_{filename.name}"
        shutil.copy(filename, image_path / new_filename)
    
    if inplace:
        shutil.rmtree(raw_dataset_path)
    
    print(f"Dataset prepared at {output_path}")

task_id = 638
task = tasks_df[tasks_df['id'] == task_id]
project_name = task['project_name']

download_annotation(task_id, export_format, f"task_{task_id}.zip.part", save_images=True)
# rename file after download
os.rename(f"task_{task_id}.zip.part", f"task_{task_id}.zip")
unzip_annotation(f"task_{task_id}.zip", f"{task_id}")
remap_annotation(f"{task_id}")
prepare_dataset(f"{task_id}", task.to_dict('records')[0])
