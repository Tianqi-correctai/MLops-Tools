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
    

def get_train_runs(model):
    runs = {}  # {path : name}
    if model == "YoloV5":
        yolov5 = Path("../backends/data/train/YoloV5")
        
        if yolov5.exists():
            for run in yolov5.iterdir():
                if run.is_dir() and (run / "run").exists():
                    runs[str((run / "run").resolve())] = run.stem

        if Path("../backends/nets/yolov5/runs/train").exists():
            for run in Path("../backends/nets/yolov5/runs/train").iterdir():
                if run.is_dir():
                    path_str = str(run.resolve())
                    if runs.get(path_str) is None:
                        runs[path_str] = path_str

        return runs

    elif model == "YoloV8":
        raise NotImplementedError
    
def get_train_weights(model):
    weights = {}  # {name : path}
    if model == "YoloV5":
        runs = get_train_runs(model)
        for run_path, run_name in runs.items():
            run_weights = Path(run_path) / "weights"
            if run_weights.exists():
                for weight in run_weights.iterdir():
                    if weight.is_file() and weight.suffix == ".pt":
                        weights[run_name + "/" + weight.name] = str(weight.resolve())
        return weights

    elif model == "YoloV8":
        return {}
    
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
            
st.title("Training Tasks")

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
    st.dataframe(df, hide_index=True, column_order=['task_id', 'model', 'status', 'data', 'command'])
else:
    st.subheader("No Current Task")
    
# display the finished tasks
if len(tasks['finished_tasks']) > 0:
    st.subheader("Finished Tasks")

    df = pd.DataFrame(tasks['finished_tasks'])
    st.dataframe(df, hide_index=True, column_order=['task_id', 'model', 'status', 'data', 'artifacts', 'command'])

# display the queued tasks
if len(tasks['queue']) > 0:
    st.subheader("Queued Tasks")
    df = pd.DataFrame(tasks['queue'])
    df[['selected']] = False
    col1, col2 = st.columns([3, 1])
    columns = list(df.columns)
    columns.remove('selected')
    edited_data = st.data_editor(df[['selected', *columns]], hide_index=True, disabled=columns, column_order=['selected', 'task_id', 'model', 'status', 'command'])
    with col2:
        if st.button("Remove Tasks", type='primary', use_container_width=True):
            for index, row in edited_data.iterrows():
                if row['selected']:
                    requests.delete(request_url + f"/remove-task/{row['task_id']}")
            st.rerun()

# add a new task
st.title("Start New Training")
st.write("Please select the model and the dataset to start training.")
datasets = get_availible_datasets(st.session_state.datasets_path)

train_indices = []
val_indices = []


model = st.selectbox("Model", ["YoloV5", "YoloV8"])
weights = get_train_weights(model)

col1, col2 = st.columns([1, 1])
with col1:
    epochs = st.number_input("Epochs", min_value=1, value=100, step=1, max_value=1000)
with col2:
    batch_size = st.number_input("Batch Size", min_value=1, value=16, step=1, max_value=1000)

options = ["Train From Scratch", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
options += list(weights.keys())
weight = st.selectbox("Pretrained Weights", options)
if weight == "Train From Scratch":
    weight = ""
elif weight in weights:
    weight = weights[weight]

with st.expander("Advanced Options"):
    extra_args = st.text_input("Extra Arguments", value="")
    remark = st.text_area("Remark", value="")

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

    ], format_func='title', grow=True, return_index=True, index=0)

train_path = None
if dataset_option == 0:
    training_datasets = get_training_files(model)
    train_path = st.selectbox('Select Dataset Config',
    (str(path.resolve()) for path in training_datasets))
    if train_path is not None:
        train_path = Path(train_path)
        with st.expander(f"**{train_path.name}**"):
            with open(train_path) as f:
                st.code(f.read(), language='yaml')
else:
    config_name = st.text_input("Config Name", value=f"new_config_{date.today().strftime('%Y_%m_%d')}")
    train_set_options = []
    for dataset in datasets:
        if not dataset.is_absolute():
            train_set_options.append(dataset.stem)
        else:
            train_set_options.append(str(dataset))

    train_set = st.multiselect("Training Datasets", train_set_options)

    val_set = st.multiselect("Validation Datasets", [dataset for dataset in train_set_options if dataset not in train_set])

    # train_indices = sac.cascader(items=[sac.CasItem(dataset.stem) for dataset in datasets],
    #                                 label="Train Set", format_func='title', multiple=True, search=True, clear=True, placeholder='Select Datasets for Training', return_index=True)

    # val_indices = sac.cascader(items=[sac.CasItem(dataset.stem, disabled=(index in train_indices)) for index, dataset in enumerate(datasets)],
    #                                 label="Validation Set", format_func='title', multiple=True, search=True, clear=True, placeholder='Select Datasets for validation', return_index=True)

col1, col2, col3 = st.columns([1, 1.5, 1])
with col1:
    button_start_training = st.button("Start Training", type="primary", use_container_width=True)

if dataset_option == 1:
    with col3:
        button_save_train_config = st.button("Save Config", use_container_width=True, disabled=(config_name == "" or len(train_set) == 0))
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
    requests.post(request_url + "/start-training", json=args)
    st.rerun()




