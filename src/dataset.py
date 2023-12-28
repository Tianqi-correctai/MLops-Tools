from pathlib import Path
import shutil
import subprocess
import zipfile
import json
import pandas as pd
from tqdm import tqdm
import os

"""
Some utility functions for preparing the dataset.
"""

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


def get_video_for_task_from_disk(self, task_name, videos_path):
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

def remap_annotation(raw_dataset_path, remap_dict):
    """
    Remap the annotation files to match the desired class IDs.
    Warning: This function will overwrite the original annotation files.

    Args:
    raw_dataset_path (str): The path of the extracted annotation files.
    remap_dict (dict): The dictionary containing the class IDs mapping.
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


def get_availible_datasets(datasets_path):
    """
    Get the list of availible datasets.
    Args:
    datasets_path (str): The path where the datasets are saved.
    """
    datasets_path = Path(datasets_path)
    datasets = []
    for dataset in datasets_path.iterdir():
        if dataset.is_dir():
            datasets.append(dataset)
    datasets += get_external_datasets(datasets_path / ".external.json")
    return datasets

def get_external_datasets(config_path):
    """
    Get the list of external datasets.
    Args:
    config_path (str): The path of the config file.
    """
    dataset_paths = []
    config_path = Path(config_path)
    if config_path.exists():
        config = json.load(open(config_path))
        for entry in config["datasets"]:
            path = Path(entry["path"])
            if path.is_dir():
                dataset_paths.append(path)
            else:
                print(f"[WARN] External dataset at {entry['path']} does not exist.")
    return dataset_paths