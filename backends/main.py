from pathlib import Path
import queue
import threading
import time
from flask import Flask, jsonify, request
import subprocess
from threading import Thread, Event
import os
import datetime
from taskmanager import TaskManager

def get_train_runs(model):
    runs = {}  # {path : name}
    if model == "YoloV5":
        yolov5 = Path("../backends/data/train/YoloV5")
        
        # get runs from the data folder
        if yolov5.exists():
            for run in yolov5.iterdir():
                if run.is_dir() and (run / "run").exists():
                    runs[str((run / "run").resolve())] = run.stem

        # get runs from the yolov5/runs/train folder
        if Path("../backends/nets/yolov5/runs/train").exists():
            for run in Path("../backends/nets/yolov5/runs/train").iterdir():
                if run.is_dir():
                    path_str = str(run.resolve())
                    # check if the run is already in the runs dict
                    if runs.get(path_str) is None:
                        runs[path_str] = path_str

        return runs

    elif model == "YoloV8":
        raise NotImplementedError
    
def get_train_weights(model):
    weights = {}  # {name : path}

    if model == "YoloV5":
        yolov5 = Path("../backends/data/train/YoloV5")
        # get uploaded weights
        if (yolov5 / "uploaded").exists():
            for weight in (yolov5 / "uploaded").iterdir():
                if weight.is_file() and weight.suffix == ".pt":
                    weights["Uploaded"+ "/" + weight.name] = str(weight.resolve())

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
    

manager = TaskManager()
manager.runner.start()
app = Flask(__name__)

# check if the server is online
@app.route('/')
def ping():
    return 'pong!'

# get the list of current running tasks
@app.route('/tasks', methods=['GET'])
def get_tasks():
    tasks = manager.get_tasks()
    return jsonify(tasks), 200

# get the current task information
@app.route('/current-task', methods=['GET'])
def get_current_task_info():
    current_task = manager.get_current_task_info()
    return jsonify(current_task), 200

# add a new task
@app.route('/train', methods=['POST'])
def start_training():
    task_data = request.get_json()
    result = manager.add_task("train", task_data)
    if result:
        return jsonify(result[0]), result[1]
    else:
        return 500

# inference 
@app.route('/inference', methods=['POST'])
def inference_api():
    task_data = request.get_json()
    result = manager.add_task("inference", task_data)
    if result:
        return jsonify(result[0]), result[1]
    else:
        return 500
    
# validate
@app.route('/validate', methods=['POST'])
def validate_api():
    task_data = request.get_json()
    result = manager.add_task("validate", task_data)
    if result:
        return jsonify(result[0]), result[1]
    else:
        return 500
    
# export
@app.route('/export', methods=['POST'])
def export_api():
    task_data = request.get_json()
    result = manager.add_task("export", task_data)
    if result:
        return jsonify(result[0]), result[1]
    else:
        return 500


# stop a running task
@app.route('/stop-training', methods=['POST'])
def stop_training():
    result = manager.stop_task()
    if result:
        return jsonify(result[0]), result[1]
    else:
        return jsonify({"status": "training stopped"}), 200
    
# remove a task from the queue
@app.route('/remove-task/<int:task_id>', methods=['DELETE'])
def remove_from_queue(task_id):
    if task_id == manager.current_task['task_id']:
        manager.stop_task()
    else:
        manager.remove_task_from_queue(task_id)
    return jsonify({"status": "task removed from queue"}), 200

# get train weights
@app.route('/train-weights/<string:model>', methods=['GET'])
def get_train_weights_api(model):
    weights = get_train_weights(model)
    return jsonify(weights), 200

# upload a weight file
@app.route('/upload-weight/<string:model>', methods=['POST'])
def upload_weight(model):
    if model == "YoloV5":
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '' or file.filename.split('.')[-1] != 'pt':
            return jsonify({"error": "No pt file provided"}), 400
        if file:
            filename = ''
            if request.args.get('file_name') is not None:
                filename = request.args.get('file_name')
            if filename == '':
                filename = file.filename
            # create the uploaded folder if it does not exist
            Path(f"../backends/data/train/{model}/uploaded").mkdir(parents=True, exist_ok=True)
            file.save(f"../backends/data/train/{model}/uploaded/{filename}")
            return jsonify({"status": "file uploaded"}), 200
        else:
            return jsonify({"error": "No file provided"}), 400
    elif model == "YoloV8":
        raise NotImplementedError

# shut down the server
@app.route('/shutdown')
def shutdown_server():
    manager.terminate()
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func:
        shutdown_func()
        return 'Server shutting down...'
    else:
        return 'Server shutdown failed', 500

if __name__ == '__main__':
    app.run()
