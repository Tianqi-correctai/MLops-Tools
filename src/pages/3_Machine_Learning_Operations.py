import json
from pathlib import Path
import time
import streamlit as st
import streamlit_antd_components as sac
from datetime import date
from Home import get_params
import pandas as pd

from dataset import get_availible_datasets

get_params()

datasets_root_path = Path(st.session_state['datasets_path']).resolve()

def get_hyperparameter_files(model):
    if model == "YoloV5":
        return Path("../backends/nets/yolov5/data/hyps").glob("*.yaml")
    elif model == "YoloV8":
        return Path("../backends/nets/ultralytics/ultralytics/cfg").glob("*.yaml")
    
def get_training_files(model):
    if model == "YoloV5":
        return Path("../backends/nets/yolov5/data").glob("*.yaml")
    elif model == "YoloV8":
        return Path("../backends/nets/ultralytics/ultralytics/cfg/datasets").glob("*.yaml")
    
def write_train_config(name, model, train_datasets, val_datasets):
    config = f"""
# Train/val sets
path: {datasets_root_path}
train: {train_datasets}
val: {val_datasets}

# Classes
names: 
  0: person
  1: car
  2: truck
  3: pipe
  4: airplane
  5: bus
  6: train
  7: toe boat
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    if model == "YoloV5":
        with open(f"../backends/nets/yolov5/data/{name}.yaml", "w") as f:
            f.write(config)
    elif model == "YoloV8":
        with open(f"../backends/nets/ultralytics/ultralytics/cfg/datasets/{name}.yaml", "w") as f:
            f.write(config)


###################################################################################################
            
st.title("MLops Tasks")

# start the backend if not started
request_url = "http://localhost:5000"
import requests
try:
    requests.get(request_url)
except:
    import subprocess
    subprocess.Popen(["python", "../backends/main.py"])

with st.spinner("Connecting to backend..."):
    while True:
        try:
            requests.get(request_url)
            break
        except:
            time.sleep(0.5)
            continue

# get the current task
tasks = []
with st.spinner("Fetching tasks..."):
    while True:
        try:
            tasks = requests.get(request_url + "/tasks").json()
            break
        except:
            time.sleep(0.5)
            continue

# display the current task
refresh = False
if tasks['current_task'] is not None:
    st.subheader("Current Task")
    col1, col2= st.columns([4, 1])

    with col2:
        if st.button("Stop", type='primary', use_container_width=True):
            requests.post(request_url + "/stop-training")
            time.sleep(1)
            st.rerun()
    
    df = pd.DataFrame([tasks['current_task']])
    st.dataframe(df, hide_index=True, column_order=['task_id', 'type', 'model', 'status', 'data', 'command'])
else:
    st.subheader("No Current Task")
    
# display the finished tasks
if len(tasks['finished_tasks']) > 0:
    st.subheader("Finished Tasks")

    df = pd.DataFrame(tasks['finished_tasks'])
    st.dataframe(df, hide_index=True, column_order=['task_id', 'type', 'model', 'status', 'data', 'artifacts', 'command'])

# display the queued tasks
if len(tasks['queue']) > 0:
    st.subheader("Queued Tasks")
    df = pd.DataFrame(tasks['queue'])
    df[['selected']] = False
    col1, col2 = st.columns([3, 1])
    columns = list(df.columns)
    columns.remove('selected')
    edited_data = st.data_editor(df[['selected', *columns]], hide_index=True, disabled=columns, column_order=['selected', 'task_id', 'type',  'model', 'status', 'command'])
    with col2:
        if st.button("Remove Tasks", type='primary', use_container_width=True):
            for index, row in edited_data.iterrows():
                if row['selected']:
                    requests.delete(request_url + f"/remove-task/{row['task_id']}")
            st.rerun()

# add a new task
st.title("Start New Task")

tab1, tab2, tab3, tab4 = st.tabs([":red[Train]", ":blue[Validation]", ":green[Inference]", ":orange[Import/Export]"])


with tab1:
    st.write("Please select the model and the dataset to start training.")
    datasets = get_availible_datasets(st.session_state.datasets_path)

    model = st.selectbox("Model", ["YoloV5", "YoloV8"], key="model-tab1")

    # /train-weights/<string:model>
    weights = requests.get(request_url + f"/train-weights/{model}").json()

    col1, col2 = st.columns([1, 1])
    with col1:
        epochs = st.number_input("Epochs", min_value=1, value=100, step=1, max_value=1000)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, value=16, step=1, max_value=1000)

    options = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
    options += list(weights.keys())
    weight = st.selectbox("Pretrained Weights", options)

    if weight in weights:
        weight = weights[weight]

    with st.expander("Advanced Options"):
        extra_args = st.text_input("Extra Arguments", value="", key="extra-args-tab1")
        remark = st.text_area("Remark", value="", key="remark-tab1")

    hyps_path = st.selectbox('Select Hyperparameters',
        (str(path.resolve()) for path in get_hyperparameter_files(model)))

    if hyps_path is not None:
        hyps_path = Path(hyps_path)
        with st.expander(f"**{hyps_path.name}**"):
            with open(hyps_path) as f:
                st.code(f.read(), language='yaml')

    dataset_option = sac.segmented(

        items=[

            sac.SegmentedItem(label='Dataset Configs', icon='collection'),

            sac.SegmentedItem(label='New Config', icon='plus-circle'),

        ], format_func='title', grow=True, return_index=True, index=0, key="dataset-option-tab1")

    train_path = None
    if dataset_option == 0:
        training_datasets = get_training_files(model)
        train_path = st.selectbox('Select Dataset Config',
        (str(path.resolve()) for path in training_datasets), key="dataset-config-tab1")
        if train_path is not None:
            train_path = Path(train_path)
            with st.expander(f"**{train_path.name}**"):
                with open(train_path) as f:
                    st.code(f.read(), language='yaml')
    else:
        config_name = st.text_input("Config Name", value=f"new_config_{date.today().strftime('%Y_%m_%d')}", key="config-name-tab1")
        val_set_options = []
        for dataset in datasets:
            if not dataset.is_absolute():
                val_set_options.append(dataset.stem)
            else:
                val_set_options.append(str(dataset))

        train_set = st.multiselect("Training Datasets", val_set_options, key="train-set-tab1")

        val_set = st.multiselect("Validation Datasets", [dataset for dataset in val_set_options if dataset not in train_set], key="val-set-tab1")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        button_start_training = st.button("Start Training", type="primary", use_container_width=True)

    if dataset_option == 1:
        with col3:
            button_save_train_config = st.button("Save Config", use_container_width=True, disabled=(config_name == "" or len(train_set) == 0), key="save-config-tab1")
            if button_save_train_config:
                write_train_config(config_name, model, train_set, val_set)

    if button_start_training:
        if dataset_option == 0:
            pass
        if dataset_option == 1:
            write_train_config(config_name, model, train_set, val_set)
            train_path = Path(f"../backends/nets/yolov5/data/{config_name}.yaml")

        args = {
            "model": model,
            "args": {
                "epoch": epochs,
                "batch-size": batch_size,
                "hyp": str(hyps_path.resolve()),
                "data": str(train_path.resolve()),
                "weights": weight,
            },
            "extra_args": extra_args,
            "remark": remark,
        }
        requests.post(request_url + "/train", json=args)
        st.rerun()

with tab2:
    st.write("Please select the model and the dataset to start validation.")
    datasets = get_availible_datasets(st.session_state.datasets_path)

    model = st.selectbox("Model", ["YoloV5", "YoloV8"], key="model-tab2")
    weights = requests.get(request_url + f"/train-weights/{model}").json()

    weight = st.selectbox("Weights", weights.keys(), key="weights-tab2")

    if weight in weights:
        weight = weights[weight]

    with st.expander("Advanced Options"):
        extra_args = st.text_input("Extra Arguments", value="", key="extra-args-tab2")
        remark = st.text_area("Remark", value="", key="remark-tab2")

    dataset_option = sac.segmented(

        items=[

            sac.SegmentedItem(label='Dataset Configs', icon='collection'),

            sac.SegmentedItem(label='New Config', icon='plus-circle'),

        ], format_func='title', grow=True, return_index=True, index=0, key="dataset-option-tab2")   

    train_path = None
    if dataset_option == 0:
        training_datasets = get_training_files(model)
        train_path = st.selectbox('Select Dataset Config',
        (str(path.resolve()) for path in training_datasets), key="dataset-config-tab2")
        if train_path is not None:
            train_path = Path(train_path)
            with st.expander(f"**{train_path.name}**"):
                with open(train_path) as f:
                    st.code(f.read(), language='yaml')
    else:
        config_name = st.text_input("Config Name", value=f"new_config_{date.today().strftime('%Y_%m_%d')}", key="config-name-tab2")
        val_set_options = []
        for dataset in datasets:
            if not dataset.is_absolute():
                val_set_options.append(dataset.stem)
            else:
                val_set_options.append(str(dataset))
        train_set = []
        val_set = st.multiselect("Validation Datasets", val_set_options, key="val-set-tab2")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        button_start_validation = st.button("Start Validation", type="primary", use_container_width=True)

    if dataset_option == 1:
        with col3:
            button_save_train_config = st.button("Save Config", use_container_width=True, disabled=(config_name == "" or len(train_set) == 0))
            if button_save_train_config:
                write_train_config(config_name, model, train_set, val_set)

    if button_start_validation:
        if dataset_option == 0:
            pass
        if dataset_option == 1:
            write_train_config(config_name, model, train_set, val_set)
            train_path = Path(f"../backends/nets/yolov5/data/{config_name}.yaml")

        args = {
            "model": model,
            "args": {
                "data": str(train_path.resolve()),
                "weights": weight,
            },
            "extra_args": extra_args,
            "remark": remark,
        }
        requests.post(request_url + "/validate", json=args)
        st.rerun()

with tab3:
    st.write("Please select the model and the video to start inference.")
    model = st.selectbox("Model", ["YoloV5", "YoloV8"])
    weights = requests.get(request_url + f"/train-weights/{model}").json()

    weight = st.selectbox("Weights", weights.keys())

    with st.expander("Advanced Options"):
        extra_args = st.text_input("Extra Arguments", value="")
        remark = st.text_area("Remark", value="")

    if weight in weights:
        weight = weights[weight]

    source_option = sac.segmented(

        items=[

            sac.SegmentedItem(label='Select Source', icon='collection'),

            sac.SegmentedItem(label='Upload New Source', icon='plus-circle'),

        ], format_func='title', grow=True, return_index=True, index=0, key="source-option-tab3")   
    
    if source_option == 0:
        # get source list
        sources = requests.get(request_url + "/sources").json()
        source = st.selectbox("Detection Source", sources.keys())

        conf_thres = st.number_input("Confidence Threshold", min_value=0.0, value=0.4, step=0.01, max_value=1.0)

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col1:
            button_start_inference = st.button("Start Inference", type="primary", use_container_width=True)

        if button_start_inference:

            args = {
                "model": model,
                "args": {
                    "source": sources[source],
                    "weights": weight,
                    "conf-thres": conf_thres,
                },
                "extra_args": extra_args,
                "remark": remark,
            }
            requests.post(request_url + "/inference", json=args)
            st.rerun()
    
    if source_option == 1:

        with st.form("my-source-upload-form", clear_on_submit=True, border=False):
            source_name = st.text_input("Upload Name", value="", key="upload-name-tab3")
            uploaded_file = st.file_uploader("Upload file", accept_multiple_files=False, )
            col1, col2, col3 = st.columns([1, 1.5, 1])
            upload_button = st.form_submit_button("Upload", type="primary", use_container_width=True)

            if upload_button:
                if (uploaded_file is not None):
                    # use /upload-weight/<string:model>
                    if source_name != "":
                        response = requests.post(request_url + f'/upload-source?file_name={source_name}', files={"file": uploaded_file})
                    else:
                        response = requests.post(request_url + f"/upload-source", files={"file": uploaded_file})
                    st.rerun()

with tab4:
    st.write("Please select the model and manage weights.")
    model = st.selectbox("Model", ["YoloV5", "YoloV8"], key="model-tab4")
    weights = requests.get(request_url + f"/train-weights/{model}").json()

    options = ["Upload New Weights"] + list(weights.keys())
    weight = st.selectbox("Weights", options, key="weights-tab4")
    if weight != "Upload New Weights":
        export_name = st.text_input("Export Name Without Extension", value="", key="export-name-tab4")

    if weight in weights:
        weight = weights[weight]

    if weight == "Upload New Weights":
        with st.form("my-upload-form", clear_on_submit=True, border=False):
            import_name = st.text_input("Upload Name Without Extension", value="", key="import-name-tab4")
            uploaded_file = st.file_uploader("Upload Weights", accept_multiple_files=False, type="pt")
            col1, col2, col3 = st.columns([1, 1.5, 1])
            upload_button = st.form_submit_button("Upload", type="primary", use_container_width=True)

            if upload_button:
                if (uploaded_file is not None):
                    # use /upload-weight/<string:model>
                    if import_name != "":
                        response = requests.post(request_url + f'/upload-weight/{model}?file_name={import_name+".pt"}', files={"file": uploaded_file})
                    else:
                        response = requests.post(request_url + f"/upload-weight/{model}", files={"file": uploaded_file})
                    st.rerun()
    else:
        with st.expander("Advanced Options"):
            extra_args = st.text_input("Extra Arguments", value="", key="extra-args-tab4")
            remark = st.text_area("Remark", value="", key="remark-tab4")
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col1:
            button_start_export = st.button("Start Export", type="primary", use_container_width=True)

        if button_start_export:

            args = {
                "model": model,
                "args": {
                    "weights": weight,
                    "include": "onnx",
                },
                "export_name": export_name+".pt",
                "extra_args": extra_args,
                "remark": remark,
            }
            requests.post(request_url + "/export", json=args)
            st.rerun()