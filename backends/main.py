from pathlib import Path
from flask import Flask, jsonify, request
from taskmanager import TaskManager

def get_train_runs(model):
    """
    Retrieve training runs for a given model.

    Parameters:
    model (str): The model name.

    Returns:
    dict: A dictionary mapping the path of each training run to its name.
    """
    runs = {}  # {path : name}
    
    # Handle YoloV5 model case
    if model == "YoloV5":
        yolov5 = Path("../backends/data/train/YoloV5")
        
        # Get runs from the data folder
        if yolov5.exists():
            for run in yolov5.iterdir():
                if run.is_dir() and (run / "run").exists():
                    runs[str((run / "run").resolve())] = run.stem

        # Get runs from the yolov5/runs/train folder
        if Path("../backends/nets/yolov5/runs/train").exists():
            for run in Path("../backends/nets/yolov5/runs/train").iterdir():
                if run.is_dir():
                    path_str = str(run.resolve())
                    if runs.get(path_str) is None:
                        runs[path_str] = path_str

        return runs
    elif model == "YoloV8":
        yolov8 = Path("../backends/data/train/YoloV8")
        
        # Get runs from the data folder
        if yolov8.exists():
            for run in yolov8.iterdir():
                if run.is_dir() and (run / "run").exists():
                    runs[str((run / "run").resolve())] = run.stem


        # Get runs from the yolov8/runs/train folder
        if Path("../backends/nets/ultralytics/runs").exists():
            for run in Path("../backends/nets/ultralytics/runs").iterdir():
                if run.is_dir() and "train" in run.name:
                    path_str = str(run.resolve())
                    if runs.get(path_str) is None:
                        runs[path_str] = path_str

        return runs        
    
def get_train_weights(model):
    """
    Retrieve training weights for a given model.

    Parameters:
    model (str): The model name.

    Returns:
    dict: A dictionary mapping the name of each weight file to its path.
    """
    weights = {}  # {name : path}

    path_to_runs = None
    if model == "YoloV5":
        path_to_runs = Path("../backends/data/train/YoloV5")
    elif model == "YoloV8":
        path_to_runs = Path("../backends/data/train/YoloV8")
    else:
        return {}
        
    # Get uploaded weights
    if (path_to_runs / "uploaded").exists():
        for weight in (path_to_runs / "uploaded").iterdir():
            if weight.is_file() and weight.suffix == ".pt":
                weights["Uploaded" + "/" + weight.name] = str(weight.resolve())

    runs = get_train_runs(model)
    for run_path, run_name in runs.items():
        run_weights = Path(run_path) / "weights"
        if run_weights.exists():
            for weight in run_weights.iterdir():
                if weight.is_file() and weight.suffix == ".pt":
                    weights[run_name + "/" + weight.name] = str(weight.resolve())

    return weights


manager = TaskManager()
manager.runner.start()

app = Flask(__name__)

@app.route('/')
def ping():
    """
    Check if the server is online.

    Returns:
    str: A simple response to indicate the server is running.
    """
    return 'pong!'


@app.route('/tasks', methods=['GET'])
def get_tasks():
    """
    Get the list of current running tasks.

    Returns:
    JSON: List of tasks with their details.
    """
    tasks = manager.get_tasks()
    return jsonify(tasks), 200

@app.route('/current-task', methods=['GET'])
def get_current_task_info():
    """
    Get the information of the current task being processed.

    Returns:
    JSON: Information of the current task.
    """
    current_task = manager.get_current_task_info()
    return jsonify(current_task), 200

@app.route('/train', methods=['POST'])
def start_training():
    """
    Add a new training task to the task manager.

    Returns:
    JSON: The result of the task addition.
    """
    task_data = request.get_json()
    result = manager.add_task("train", task_data)
    return jsonify(result[0]), result[1] if result else 500

@app.route('/inference', methods=['POST'])
def inference_api():
    """
    Add a new inference task to the task manager.

    Returns:
    JSON: The result of the task addition.
    """
    task_data = request.get_json()
    result = manager.add_task("inference", task_data)
    return jsonify(result[0]), result[1] if result else 500

@app.route('/validate', methods=['POST'])
def validate_api():
    """
    Add a new validation task to the task manager.

    Returns:
    JSON: The result of the task addition.
    """
    task_data = request.get_json()
    result = manager.add_task("validate", task_data)
    return jsonify(result[0]), result[1] if result else 500

@app.route('/export', methods=['POST'])
def export_api():
    """
    Add a new export task to the task manager.

    Returns:
    JSON: The result of the task addition.
    """
    task_data = request.get_json()
    result = manager.add_task("export", task_data)
    return jsonify(result[0]), result[1] if result else 500

@app.route('/stop-training', methods=['POST'])
def stop_training():
    """
    Stop a running training task.

    Returns:
    JSON: Status of the stop operation.
    """
    result = manager.stop_task()
    return jsonify(result[0]), result[1] if result else jsonify({"status": "training stopped"}), 200

@app.route('/remove-task/<int:task_id>', methods=['DELETE'])
def remove_from_queue(task_id):
    """
    Remove a task from the queue.

    Parameters:
    task_id (int): The ID of the task to be removed.

    Returns:
    JSON: Status of the removal operation.
    """
    if task_id == manager.current_task['task_id']:
        manager.stop_task()
    else:
        manager.remove_task_from_queue(task_id)
    return jsonify({"status": "task removed from queue"}), 200

@app.route('/train-weights/<string:model>', methods=['GET'])
def get_train_weights_api(model):
    """
    Get training weights for a specified model.

    Parameters:
    model (str): The model name.

    Returns:
    JSON: Dictionary of model weights.
    """
    weights = get_train_weights(model)
    return jsonify(weights), 200

@app.route('/upload-weight/<string:model>', methods=['POST'])
def upload_weight(model):
    """
    Upload a weight file for a specified model.

    Parameters:
    model (str): The model name.

    Returns:
    JSON: Status of the upload operation.
    """
    if model == "YoloV5" or model == "YoloV8":
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
    else:
        raise NotImplementedError

@app.route('/upload-source', methods=['POST'])
def upload_video():
    """
    Upload a detection source (e.g., video file).

    Returns:
    JSON: Status of the upload operation.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file provided"}), 400
    if file:
        filename = ''
        if request.args.get('file_name') is not None:
            filename = request.args.get('file_name')
        if filename == '':
            filename = file.filename
        # create the uploaded folder if it does not exist
        Path(f"../backends/data/sources").mkdir(parents=True, exist_ok=True)
        # unzip if it is a zip file
        if filename.split('.')[-1] == 'zip':
            from zipfile import ZipFile
            with ZipFile(file, 'r') as zipObj:
                zipObj.extractall(f"../backends/data/sources/{filename.split('.')[0]}")
        else:
            file.save(f"../backends/data/sources/{filename}")
        return jsonify({"status": "file uploaded"}), 200
    else:
        return jsonify({"error": "No file provided"}), 400
    
@app.route('/sources', methods=['GET'])
def get_videos():
    """
    Get a list of all uploaded detection sources.

    Returns:
    JSON: List of sources.
    """
    sources = {}
    source_folder = Path("../backends/data/sources")
    if source_folder.exists():
        for source in source_folder.iterdir():
            sources[source.name] = str(source.resolve())
    return jsonify(sources), 200

@app.route('/shutdown', methods=['GET'])
def shutdown_server():
    """
    Shut down the server.

    Returns:
    str: A message indicating the shutdown status.
    """
    manager.terminate()
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func:
        shutdown_func()
        return 'Server shutting down...'
    else:
        return 'Server shutdown failed', 500

if __name__ == '__main__':
    app.run()
