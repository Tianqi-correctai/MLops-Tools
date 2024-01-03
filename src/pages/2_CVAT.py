import json
import os
import pandas as pd
import streamlit as st
import humanize

from cvat_api import CVATAPI
from dataset import prepare_dataset, remap_annotation, unzip_annotation


# Parameters
json_file = open("config.json")
params = json.load(json_file)

cvat_url = params['cvat_url']
username = params['username']
password = params['password']
datasets_path = params['datasets_path']
export_format = params['export_format']
remap_dict = params['remap_dict']


# Initialize session state variables
if 'CVAT' not in st.session_state:
    st.session_state['CVAT'] = None
    st.session_state['projects'] = None
    st.session_state['tasks'] = None

st.title("CVAT Dataset Management")

# Authenticate and fetch data only if not done already
if st.session_state['CVAT'] is None:
    with st.spinner('Authenticating...'):
        st.session_state['CVAT'] = CVATAPI(cvat_url)
        st.session_state['CVAT'].authenticate(username, password)


# crete two columns
col1, col2 = st.columns([3, 1])

with col1:
    st.write("Authenticated!")   
with col2:
    # Go to cvat site button
    st.link_button("Switch to CVAT", url=cvat_url, type='primary',use_container_width=True)


if st.session_state['projects'] is None:
    with st.spinner('Fetching projects...'):
        st.session_state['projects'] = st.session_state['CVAT'].fetch_all_pages(f"{cvat_url}/api/projects")
        st.session_state.projects_df = pd.DataFrame(st.session_state['projects'])

        # convert time to human readable
        st.session_state.projects_df['created_date'] = st.session_state.projects_df['created_date'].apply(lambda x: humanize.naturaltime(pd.to_datetime(x)))
        st.session_state.projects_df['updated_date'] = st.session_state.projects_df['updated_date'].apply(lambda x: humanize.naturaltime(pd.to_datetime(x)))
        # get task count
        st.session_state.projects_df['task_count'] = st.session_state.projects_df['tasks'].apply(lambda x: x['count'])
        # get a preview of the project
        st.session_state.projects_df['preview'] = st.session_state.projects_df['id'].apply(lambda x: st.session_state['CVAT'].get_project_preview(x))
        st.session_state.projects_df['selected'] = [False] * st.session_state.projects_df.shape[0]

    # list Tasks
if st.session_state['tasks'] is None:
    with st.spinner('Fetching tasks...'):
        st.session_state['tasks'] = st.session_state['CVAT'].fetch_all_pages(f"{cvat_url}/api/tasks")

        st.session_state.tasks_df = pd.DataFrame(st.session_state['tasks'])
        st.session_state.tasks_df['project_name'] = st.session_state.tasks_df['project_id'].map(st.session_state.projects_df.set_index('id')['name'])
        # convert time to human readable
        st.session_state.tasks_df['created_date'] = st.session_state.tasks_df['created_date'].apply(lambda x: humanize.naturaltime(pd.to_datetime(x)))
        st.session_state.tasks_df['updated_date'] = st.session_state.tasks_df['updated_date'].apply(lambda x: humanize.naturaltime(pd.to_datetime(x)))
        st.session_state.tasks_df['selected'] = [False] * st.session_state.tasks_df.shape[0]


# project table
st.write("**Projects:**")
project_editor = st.data_editor(
    st.session_state.projects_df[['selected', 'id', 'preview', 'name', 'task_count', 'created_date', 'updated_date', 'status']],
    column_config={
        "preview": st.column_config.ImageColumn(  # preview image
            "Preview Image", help="Preview Image"
        ),
        'task_count': st.column_config.NumberColumn(
            "Tasks", help="Task Count"
        ),
        "selected": st.column_config.CheckboxColumn(  # select box
            "Select",
            help="Select projects to download",
            default=False,
        )
    },
    hide_index=True,
    disabled=['id', 'preview', 'name', 'task_count', 'created_date', 'updated_date', 'status']
)

# When a project is selected, select all tasks in that project
if st.session_state['tasks'] is None:
    pass
else:
    st.session_state.tasks_df['selected'] = [False] * st.session_state.tasks_df.shape[0]
    for project_id in project_editor[project_editor['selected'] == True]['id']:
        st.session_state.tasks_df['selected'] = st.session_state.tasks_df['selected'] | (st.session_state.tasks_df['project_id'] == project_id)


# task table
st.write("**Tasks:**")
task_editor = st.data_editor(
    st.session_state.tasks_df[['selected', 'id', 'project_name', 'name', 'size', 'created_date', 'updated_date', 'status']],
    hide_index=True,
    disabled=['id', 'project_name', 'name', 'size', 'created_date', 'updated_date', 'status'],
    column_config={
        "selected": st.column_config.CheckboxColumn(
            "Select",
            help="Select tasks to download",
            default=False,
        )
    },
)

# Download selected tasks
button = st.button("Download Selected Tasks", disabled=task_editor[task_editor['selected'] == True].shape[0] == 0)

if button:
    with st.spinner('Downloading... Do not close this page!'):
        my_bar = st.progress(0, text="Downloading...")
        all_size = task_editor[task_editor['selected'] == True]['size'].sum()
        #st.write(task_editor[task_editor['selected'] == True][['id', 'size']])
        downloaded_size = 0
        for task_id in task_editor[task_editor['selected'] == True]['id']:
            my_bar.progress(downloaded_size / all_size, text=f"Downloading task {task_id}...")
            # check if we already have this task in the dataset by the metafiles
            meta = pd.DataFrame(st.session_state.tasks)[st.session_state.tasks_df['id'] == task_id]
            project_name = st.session_state.tasks_df[st.session_state.tasks_df['id'] == task_id]['project_name'].values[0]
            meta["project_name"] = project_name

            if os.path.exists(f"{datasets_path}/{project_name}/.meta/{task_id}.json"):
                # check if the updated date is the same
                with open(f"{datasets_path}/{project_name}/.meta/{task_id}.json") as f:
                    meta_loaded = json.load(f)
                    if meta_loaded['updated_date'] == meta['updated_date'].values[0]:
                        print("Skipping task (up to date)", task_id)
                        downloaded_size += st.session_state.tasks_df[st.session_state.tasks_df['id'] == task_id]['size'].values[0]
                        continue

            # check if we have already downloaded this task
            if os.path.exists(f"{datasets_path}/task_{task_id}.zip"):
                print("Skipping download task (already downloaded)", task_id)
            else:
                # download the task
                st.session_state['CVAT'].download_annotation(task_id, export_format, f"{datasets_path}/task_{task_id}.zip.part", save_images=True)
                # rename file after download
                os.rename(f"{datasets_path}/task_{task_id}.zip.part", f"{datasets_path}/task_{task_id}.zip")

            # unzip the downloaded task
            unzip_annotation(f"{datasets_path}/task_{task_id}.zip", f"{datasets_path}/{task_id}")

            # remap the annotations with the remap_dict
            remap_annotation(f"{datasets_path}/{task_id}", remap_dict)

            # prepare the dataset
            prepare_dataset(f"{datasets_path}/{task_id}", meta.to_dict('records')[0])

            # update the progress
            downloaded_size += st.session_state.tasks_df[st.session_state.tasks_df['id'] == task_id]['size'].values[0]

        my_bar.progress(1.0, text="Done!")
