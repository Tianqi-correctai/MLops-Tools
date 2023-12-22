# Correct-AI MLOps Tools

## Overview

The Correct-AI MLOps Tools is a collection of tools designed to streamline the process of machine learning projects. These tools provide functionalities for exporting annotated datasets in various formats, remapping class IDs, and managing CVAT instances.

## Features

- Export annotated datasets in different formats: The tools allow you to export annotated datasets in formats such as YOLO 1.1, making it easier to integrate the datasets into your machine learning pipelines.
- Class ID remapping: You can easily remap class IDs to match your specific requirements. The tools provide a dictionary where you can define the mapping between class names and IDs.
- CVAT instance management: The tools provide an interface to interact with CVAT instances, allowing you to authenticate, retrieve parameters, and perform other operations.

## Getting Started

To use the Correct-AI MLOps Tools, follow these steps:

1. Install the required dependencies: Make sure you have the necessary dependencies installed, including the CVAT API, Streamlit, and pandas.
2. Set up CVAT parameters: Replace the placeholder values in the code with your CVAT URL, username, password, export format, and datasets path.
3. Instantiate the CVAT API: Uncomment the lines of code that instantiate the CVAT API and authenticate with your CVAT credentials.
4. Run the main function: Uncomment the line of code that calls the main function to start the Correct-AI MLOps Tools.

---

## User Manual

### **1. Dataset Management Page**

- Overview of datasets
- Tree view of datasets
- Select and delete datasets
- Extract subsets of datasets
- Display images with bounding box annotations

#### Dataset Overview (Overview Tab)

The first tab shows the **overview** of all the datasets on this server. You can see the **amount of images** and **task meta data** in each dataset project.

#### Tree View (Manage Tab)

The **tree view** on the left side of the page displays the datasets in a hierarchical structure. You can expand and collapse the tree nodes to navigate through the datasets.

#### Select and Delete Datasets

To delete a dataset, select it from the tree view by checking the **checkbox** next to its name. Then, click the "**Delete Selected**" button. This will remove the selected datasets from the system.

#### Extract Subsets of Datasets

You can **extract subsets** of datasets by **selecting images** and **specifying the split rate**. To select images, check the **checkboxes** next to their names in the **tree view**. Then, click the "**Extract Selected**" button. You will be prompted to enter the **split rate** and provide a **name** for the new dataset. If the **split rate** is set to **100%**, the **entire** selected dataset will be extracted as a new dataset. Otherwise, **a random subset** of images will be extracted based on the **specified split rate** and create **the rest subset**.

#### Display Images with Bounding Box Annotations (Frames Tab)

If there are selected images, they will be **displayed** in the main area of the page. You can use the **slider** to navigate through the images. Each image is shown with its corresponding bounding box **annotations**. The annotations are loaded from the label files associated with the images.

If no images are selected, a message will be displayed indicating that no images are selected.

### **2. CVAT Page**

This page is responsible for managing projects and tasks in the CVAT application. It provides a user-friendly interface for selecting and downloading tasks.

#### Projects

The **Projects** section displays a table of projects with the following columns:

- **Select**: Checkbox column to select projects for download.
- **ID**: Project ID.
- **Preview**: Preview image of the project.
- **Name**: Name of the project.
- **Tasks**: Number of tasks in the project.
- **Created Date**: Date when the project was created.
- **Updated Date**: Date when the project was last updated.
- **Status**: Status of the project.

You can select one or more projects by checking the corresponding checkboxes. When a project is selected, all tasks within that project will be automatically selected.

#### Tasks

The **Tasks** section displays a table of tasks with the following columns:

- **Select**: Checkbox column to select tasks for download.
- **ID**: Task ID.
- **Project Name**: Name of the project to which the task belongs.
- **Name**: Name of the task.
- **Size**: Size of the task.
- **Created Date**: Date when the task was created.
- **Updated Date**: Date when the task was last updated.
- **Status**: Status of the task.

You can select one or more tasks by checking the corresponding checkboxes.

#### Download Selected Tasks

Clicking the **Download Selected Tasks** button will initiate the download process for the selected tasks. The progress of the download will be displayed in a progress bar. The tasks will be downloaded as ZIP files and saved in the specified dataset path.

During the download process, the following steps will be performed for each selected task:

1. Check if the task is already up to date by comparing the updated date with the existing metafiles.
2. If the task is up to date, skip the download and move to the next task.
3. If the task has not been downloaded before, download it using the CVAT API and save it as a ZIP file.
4. Unzip the downloaded task to the dataset path.
5. Remap the annotations of the task using the provided remap dictionary.
6. Prepare the dataset for further processing.
7. Update the progress bar with the downloaded size.

Once all selected tasks have been downloaded and processed, the progress bar will reach 100% and display "Done!".

Please note that this page requires access to the CVAT API and the specified dataset path for downloading and processing tasks.  
