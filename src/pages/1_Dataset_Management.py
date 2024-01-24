import PIL
import streamlit as st
from dataset import *
from Home import get_params
from pathlib import Path
import streamlit_antd_components as sac
import cv2
import random

# List of supported image formats
IMG_FORMATS = '.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'

# Getting parameters
get_params()

# Setting page configuration
st.set_page_config(layout="wide")
st.title("Available Datasets")

# Getting available datasets
datasets = get_availible_datasets(st.session_state.datasets_path)
dataset_indexes = {}

dataset_images = {}
selected_images = []

def build_tree_project(datasets):
    """
    Builds a tree structure for the given datasets.

    Args:
        datasets (list): A list of dataset paths.

    Returns:
        list: A tree structure representing the datasets.
    """
    tree = []
    
    for i, dataset in enumerate(datasets):
        tasks = [json.load(open(json_file)) for json_file in sorted(list((dataset / ".meta").glob("*.json")))]
        children = []
        if tasks == []:
            children = build_tree_folder(dataset)
        else:
            children = build_tree_task(dataset, tasks)

        tree.append(children)
        dataset_indexes[dataset.stem] = i

    return tree

def build_tree_task(dataset, tasks):
    """
    Builds a tree structure for the given dataset and tasks.

    Args:
        dataset (Path): The dataset path.
        tasks (list): A list of tasks.

    Returns:
        list: A tree structure representing the tasks.
    """
    task_lens = []

    project_images = [file for file in Path(dataset / "images").glob(f"*") if file.suffix.lower() in IMG_FORMATS]

    dataset_display = None
    if dataset.is_absolute():
        dataset_display = str(dataset)
    else:
        dataset_display = dataset.stem


    dataset_images[dataset_display] = set(project_images)



    for j, task in enumerate(tasks):
        project_id = task["project_id"]
        task_id = task["id"]
        prefix = f"{project_id}_{task_id}_"
        task_images = [file for file in Path(dataset / "images").glob(f"{prefix}*") if file.suffix.lower() in IMG_FORMATS]
        task_lens.append(len(task_images))


        dataset_images[f"{dataset_display}#{task_id}"] = set(task_images)



    nodes = [sac.TreeItem(f"{dataset_display}#{task['id']}", icon="stack", tag=f"{task_lens[task_index]}", tooltip=task["name"]) 
                for task_index, task in enumerate(tasks)]
    
    if len(project_images) != sum(task_lens):
        nodes.append(sac.TreeItem(f"{dataset_display}#Others", icon="layers-half", tag=f"{len(project_images) - sum(task_lens)}"))
        dataset_images[f"{dataset_display}#Others"] = set(project_images) - set.union(*[dataset_images[f"{dataset_display}#{task['id']}"] for task in tasks])
    
    return sac.TreeItem(dataset_display, icon="folder2", tag=f"{len(project_images)}", children=nodes, tooltip=str(dataset))
    
    

def build_tree_folder(dataset):
    """
    Builds a tree structure for the given dataset.

    Args:
        dataset (Path): The dataset path.

    Returns:
        sac.TreeItem: A tree item representing the dataset.
    """
    tasks_images = [file for file in Path(dataset / "images").glob(f"*") if file.suffix.lower() in IMG_FORMATS]

    dataset_display = None
    if dataset.is_absolute():
        dataset_display = str(dataset)
    else:
        dataset_display = dataset.stem

    dataset_images[dataset_display] = set(tasks_images)
    return sac.TreeItem(dataset_display, icon="layers-half", tag=f"{len(tasks_images)}", tooltip=str(dataset))

@st.cache_data
def getSelectedFrames(selected: list):
    """
    Get the selected frames from the dataset.

    Args:
        selected (list): A list of selected dataset items.

    Returns:
        list: A list of selected frames.
    """
    frames = set()

    full_projects = []
    for item in selected:
        parts = item.split('#')
        if len(parts) == 1:
            full_projects.append(item)
        else:
            if parts[0] in full_projects:
                continue
        frames = frames.union(set(dataset_images[item]))
    frames = sorted(list(frames))

    return frames


def extract_dataset(image_paths, dataset_name):
    """
    Extracts a dataset by copying images and labels to a new directory.

    Args:
        image_paths (list): A list of image paths to be copied.
        dataset_name (str): The name of the new dataset.

    Returns:
        None
    """
    # create directory
    new_dataset_path = Path(st.session_state.datasets_path) / dataset_name
    new_dataset_path.mkdir(parents=True, exist_ok=True)
    # copy images
    new_dataset_images_path = new_dataset_path / "images"
    new_dataset_images_path.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        try:
            shutil.copy(image_path, new_dataset_images_path / image_path.name)
        except shutil.SameFileError:
            continue
    # copy labels
    new_dataset_labels_path = new_dataset_path / "labels"
    new_dataset_labels_path.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        label_path = Path(image_path.parent.parent / "labels" / image_path.stem).with_suffix(".txt")
        if label_path.exists():
            try:
                shutil.copy(label_path, new_dataset_labels_path / label_path.name)
            except shutil.SameFileError:
                continue

# Create tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Manage", "Frames"])

# Tab1: Overview of the datasets and their metadata 
with tab1:
    with st.container():
        for dataset in datasets:

            title = None
            if dataset.is_absolute():
                title = str(dataset)
            else:
                title = dataset.stem

            with st.expander(f"**{title}**"):
                st.write("Images:", len([file for file in list((dataset / "images").glob("*")) if file.suffix.lower() in IMG_FORMATS]))
                for json_file in sorted(list((dataset / ".meta").glob("*.json"))):
                    container = st.container(border=True)
                    container.write(f"**{json_file.stem}**")
                    container.write("Metadata:")
                    container.json(json.load(open(json_file)), expanded=False)

# Tab2: Manage the datasets, delete, extract, etc. Select for frames.
with tab2:
    col1, col2 = st.columns([3, 1])
    selected = []
    with col1:
        selected = sac.tree(items=build_tree_project(datasets), index=[], format_func=lambda x:x.split('#')[-1], icon='table', open_all=True, checkbox=True)
    
    selected_images = getSelectedFrames(selected)
    
    with col2:

        button = st.button("Delete Selected", type="primary", disabled=len(selected_images) == 0)

        if button:
            # remove projects
            for item in selected:
                parts = item.split('#')
                if len(parts) == 1:
                    dataset_path = datasets[dataset_indexes[item]]
                    shutil.rmtree(dataset_path)

                # remove meta
                if len(parts) == 2:
                    dataset_path = datasets[dataset_indexes[parts[0]]]
                    task_id = parts[1]
                    if task_id != "Others":
                        meta_path = dataset_path / ".meta" / f"{task_id}.json"
                        if meta_path.exists():
                            os.remove(meta_path)

            # remove tasks images, labels
            for image_path in selected_images:
                label_path = Path(image_path.parent.parent / "labels" / image_path.stem).with_suffix(".txt")
                if image_path.exists():
                    os.remove(image_path)
                if label_path.exists():
                    os.remove(label_path)

            selected = []
            selected_images = []
            st.rerun()

        button2 = st.button("Extract Selected", type="secondary", disabled=len(selected_images) == 0)
        if button2:
            st.session_state.extract_button = True

        if st.session_state.get("extract_button", False):
            # input a percentage of the dataset to extract
            percentage = st.number_input("Split Rate", min_value=0, max_value=100, value=100, step=1)
            # input a name for the new dataset
            new_dataset_name = st.text_input("New dataset name", value="new_dataset")
            new_dataset_name_alt = st.text_input("Rest dataset name", value="new_dataset_rest", disabled=percentage == 100)
            # extract the dataset
            extract_button = st.button("Extract", type="primary")

            if extract_button:
                if percentage == 100:
                    extract_dataset(selected_images, new_dataset_name)
                else:
                    random_images = random.sample(getSelectedFrames(selected), int(len(selected_images) * percentage / 100))
                    extract_dataset(random_images, new_dataset_name)
                    extract_dataset(set(selected_images) - set(random_images), new_dataset_name_alt)

                st.session_state.extract_button = False
                st.rerun()
        
    
    st.write("Selected:", len(selected_images), selected)

def draw_bbox(image, coords, class_id):
    """
    Draw bounding box on the image.

    Parameters:
    - image: The image to draw the bounding box on.
    - coords: The coordinates of the bounding box.
    - class_id: The class ID of the bounding box.

    Returns:
    - The image with the bounding box drawn.
    """
    # convert coords to cv2 format
    # from <center x> <center y> <width> <height> to pt1 = (x, y), pt2 = (x + width, y + height)
    # use image.shape to get image width and height
    
    pt1 = (int(float(coords[0]) * image.shape[1] - float(coords[2]) * image.shape[1] / 2), int(float(coords[1]) * image.shape[0] - float(coords[3]) * image.shape[0] / 2))
    pt2 = (int(float(coords[0]) * image.shape[1] + float(coords[2]) * image.shape[1] / 2), int(float(coords[1]) * image.shape[0] + float(coords[3]) * image.shape[0] / 2))
    

    # use different colors for different classes
    colors = {
        '0': (255, 0, 0),
        '1': (0, 255, 0),
        '2': (0, 0, 255),
        '3': (255, 255, 0),
        '4': (255, 0, 255),
        '5': (0, 255, 255),
        '6': (255, 255, 255),
        '7': (0, 0, 0),
        '8': (128, 128, 128),
        '9': (128, 0, 0),
        '10': (0, 128, 0),
        '11': (0, 0, 128),
        '12': (128, 128, 0),
        '13': (128, 0, 128),
        '14': (0, 128, 128),
        '15': (192, 192, 192),
        '16': (128, 128, 128),
        '17': (153, 153, 255),
        '18': (153, 51, 102),
        '19': (255, 255, 204),
        '20': (204, 255, 255),
        '21': (102, 0, 102),
        '22': (255, 128, 128),
        '23': (0, 102, 204),
        '24': (204, 204, 255),
        '25': (0, 0, 128),
        '26': (255, 0, 255),
        '27': (255, 255, 0),
        '28': (0, 255, 255),
        '29': (128, 0, 128),
        '30': (128, 0, 0),
    }
    
    # draw bbox
    cv2.rectangle(image, pt1, pt2, colors[class_id], 2)

    return image

# Tab3: View the selected frames, and their annotations
with tab3:

    if len(selected_images) > 0:
        index = st.select_slider('Images from dataset', options=range(len(selected_images)))

        image = cv2.imread(str(selected_images[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # load annotation
        file_name = ".".join(str(selected_images[index].resolve()).split("/")[-1].split("."))
        annotation_path = Path(selected_images[index].parent.parent / "labels" / file_name).with_suffix(".txt")
        #print(annotation_path)
        #annotation_path = Path(selected_images[index].parent.parent / "labels" / selected_images[index].stem).with_suffix(".txt")
        if annotation_path.exists():
            annotations = annotation_path.read_text().split('\n')

            # add annotation overlay
            for annotation in annotations:
                if annotation == "":
                    continue
                class_id, *coords = annotation.split()
                image = draw_bbox(image, coords, class_id)

        st.image(image, caption=selected_images[index].resolve(), use_column_width=True)

    else:
        st.write("No images selected")
