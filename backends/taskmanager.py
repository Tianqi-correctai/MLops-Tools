from pathlib import Path
import queue
import threading
import time
import subprocess
from threading import Thread, Event
import os
import datetime

ORIG_PATH = os.getcwd() # Store the original working directory
class TaskManager:
    def __init__(self):
        """
        Initialize the TaskManager class.
        """
        self.task_max_id = 1  # Tracks the ID for the next task to be added
        self.finished_tasks = []  # Stores completed tasks
        self.current_task = None  # Holds the currently executing task
        self.task_queue = queue.Queue()  # Queue for holding tasks
        self.stop_event = Event()  # Event to signal the termination of task processing
        self.stop_current_task = Event()  # Event to signal stopping the current task
        self.runner = Thread(target=self.run)  # Thread for running tasks
        self.yolo_runs_map = {}  # Map to store paths of YoloV5 runs

    def terminate(self):
        """
        Terminate the task processing thread.
        """
        self.stop_event.set()
        self.runner.join()

    def stop_task(self):
        """
        Stop the currently running task.

        Returns:
        dict: Status message with error or success information.
        """
        if self.current_task is None:
            return {"error": "No task is running"}, 400
        self.stop_current_task.set()

    def empty_tasks(self):
        """
        Empty the task queue.
        """
        self.task_queue = queue.Queue()

    def remove_task_from_queue(self, task_id):
        """
        Remove a task from the queue.

        Parameters:
        task_id (int): The ID of the task to be removed.
        """
        tasks = []
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task[0] != task_id:
                tasks.append(task)
        for task in tasks:
            self.task_queue.put(task)

    def log_handler(self, task_type, process, log_file_path):
        """
        Handle logging of task output.

        Parameters:
        task_type (str): The type of the task (e.g., 'train', 'validate').
        process (subprocess.Popen): The process running the task.
        log_file_path (str): Path to the log file.
        """
        with open(log_file_path, 'w') as log_file:
            r_pos = None
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    if "it/s]" in output or "s/it]" in output:
                        if r_pos is not None:
                            log_file.seek(r_pos)
                            log_file.truncate()
                        r_pos = log_file.tell()

                    elif r_pos is not None:
                        r_pos = None

                    if task_type == "train":
                        # get the run id ("...nets/yolov5/runs/train/exp%run_id%....")
                        if self.yolo_runs_map.get(log_file_path) is None and "Plotting labels to" in output:  # yolov5
                            run_id = output.split("Plotting labels to ")[1].split("/labels.jpg")[0]
                            self.yolo_runs_map[log_file_path] = run_id
                        elif self.yolo_runs_map.get(log_file_path) is None and "Logging results to [1m" in output:
                            run_id = output.split("Logging results to [1m")[1].split("[0m")[0]
                            self.yolo_runs_map[log_file_path] = run_id

                    if task_type == "validate" or task_type == "inference" or task_type == "export":
                        # get the run id ("...nets/yolov5/runs/train/exp%run_id%....")
                        if self.yolo_runs_map.get(log_file_path) is None and "Results saved to [1m" in output:
                            run_id = output.split("Results saved to [1m")[1].split("[0m")[0]
                            self.yolo_runs_map[log_file_path] = run_id

                    log_file.write(output)
                    log_file.flush()

    # The 'run', 'train', 'validate', 'inference', and 'export' methods follow
    # a similar pattern: they process tasks from the queue and handle task
    # execution, logging, and status management.
    def run(self):
        """
        Continuously run tasks from the task queue.
        """
        while not self.stop_event.is_set():
            try:
                task_id, task_type, task_data = self.task_queue.get(timeout=1)  # Wait for a task
                # Here, start the actual training task

                if task_type == "train":
                    task = self.train(task_id, task_data)

                if task_type == "validate":
                    task = self.validate(task_id, task_data)
                
                if task_type == "inference":
                    task = self.inference(task_id, task_data)

                if task_type == "export":
                    task = self.export(task_id, task_data)

                # move on to the next task
                self.task_queue.task_done()
                self.finished_tasks.append(task)
                self.current_task = None
                
            except queue.Empty:
                continue 

    def train(self, task_id, task_data):
        """
        Handle the training task.

        Parameters:
        task_id (int): The ID of the task.
        task_data (dict): Data associated with the task.

        Returns:
        dict: The task with its updated status.
        """
        args = []
        venv_python = None
        run_path = None

        if task_data['model'] == 'YoloV5':
            for key, value in task_data['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')       

            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/yolov5/venv/bin/python'
            run_path = Path(f"data/train/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        if task_data['model'] == 'YoloV8':
            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/ultralytics/venv/bin/python'
            run_path = Path(f"data/train/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
            
        if task_data['model'] == 'YoloV5':
            process_str = [venv_python, '-u', 'nets/yolov5/train.py', *args, *extra_args]
        elif task_data['model'] == 'YoloV8':
            save_dir = Path(__file__).resolve().parent / "nets/ultralytics/runs"
            script = f"""
from ultralytics import YOLO
model = YOLO('{task_data["args"]["weights"]}')
results = model.train(data='{task_data["args"]["data"]}', batch={task_data["args"]["batch-size"]}, epochs={task_data["args"]["epoch"]}, imgsz=640, project='{save_dir}')
print(results)
"""
            process_str = [venv_python, '-u', '-c', script]
        else:
            raise ValueError(f"Model {task_data['model']} not found")
                    
        cmd_file = run_path / 'meta.log'
        with open(cmd_file, 'w') as f:
            f.write('Command:\n')
            f.write(' '.join(process_str)+'\n\n')
            f.write('Configs:\n')
                        # get files names in process_str
            for index, arg in enumerate(process_str):
                            # if the argument is a yaml file, get the file name and previous argument name
                if arg.endswith('.yaml'):
                    f.write(f'{process_str[index-1]}: {arg}\n')
                    with open(arg, 'r') as yaml_file:
                        f.write(yaml_file.read())
                        f.write('#' * 50 + '\n')
            if task_data.get('weights') is not None:
                f.write(f'Weights: {task_data["args"]["weights"]}\n')

            if task_data.get('remark') is not None:
                f.write(f'Remark: {task_data["remark"]}\n')

        task = {
                        'task_id': task_id,
                        'type': 'train',
                        'status': "running",
                        'data': str(run_path.resolve()),
                        'command': ' '.join(process_str),
                        'model': task_data['model'],
                    }
        self.current_task = task

        process = subprocess.Popen(process_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_thread = threading.Thread(target=self.log_handler, args=("train", process, log_file))
        log_thread.start()

        stopped = False
        while process.poll() is None:
            time.sleep(1)
            if self.stop_current_task.is_set():
                self.stop_current_task.clear()
                process.kill()
                stopped = True
                break

        log_thread.join()  # Ensure the logging thread has finished

        if process.poll() == 0:
            status = "finished"
        elif stopped:
            status = "stopped"
        else:
            status = "failed"
        task['status'] = status

        # add run files to the task
        if task['model'] == 'YoloV5' or task['model'] == 'YoloV8':
            yolo_run_folder = self.yolo_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    def validate(self, task_id, task_data):
        """
        Handle the validation task.

        Parameters:
        task_id (int): The ID of the task.
        task_data (dict): Data associated with the task.

        Returns:
        dict: The task with its updated status.
        """
        args = []
        venv_python = None
        run_path = None

        if task_data['model'] == 'YoloV5':
            for key, value in task_data['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')    

            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/yolov5/venv/bin/python'
            run_path = Path(f"data/val/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        if task_data['model'] == 'YoloV8':
            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/ultralytics/venv/bin/python'
            run_path = Path(f"data/val/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
            
        if task_data['model'] == 'YoloV5':
            process_str = [venv_python, '-u', 'nets/yolov5/val.py', *args, *extra_args]
        elif task_data['model'] == 'YoloV8':
            save_dir = Path(__file__).resolve().parent / "nets/ultralytics/runs"
            script = f"""
from ultralytics import YOLO
model = YOLO('{task_data["args"]["weights"]}')
results = model.val(data='{task_data["args"]["data"]}', project='{save_dir}')
print(results)
"""
            process_str = [venv_python, '-u', '-c', script]
        else:
            raise ValueError(f"Model {task_data['model']} not found")

        cmd_file = run_path / 'meta.log'
        with open(cmd_file, 'w') as f:
            f.write('Command:\n')
            f.write(' '.join(process_str)+'\n\n')
            f.write('Configs:\n')
                        # get files names in process_str
            for index, arg in enumerate(process_str):
                            # if the argument is a yaml file, get the file name and previous argument name
                if arg.endswith('.yaml'):
                    f.write(f'{process_str[index-1]}: {arg}\n')
                    with open(arg, 'r') as yaml_file:
                        f.write(yaml_file.read())
                        f.write('#' * 50 + '\n')
            if task_data.get('weights') is not None:
                f.write(f'Weights: {task_data["args"]["weights"]}\n')
            if task_data.get('remark') is not None:
                f.write(f'Remark: {task_data["remark"]}\n')

        task = {
                        'task_id': task_id,
                        'type': 'validate',
                        'status': "running",
                        'data': str(run_path.resolve()),
                        'command': ' '.join(process_str),
                        'model': task_data['model'],
                    }
        self.current_task = task

        process = subprocess.Popen(process_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_thread = threading.Thread(target=self.log_handler, args=("validate", process, log_file))
        log_thread.start()

        stopped = False
        while process.poll() is None:
            time.sleep(1)
            if self.stop_current_task.is_set():
                self.stop_current_task.clear()
                process.kill()
                stopped = True
                break

        log_thread.join()  # Ensure the logging thread has finished

        if process.poll() == 0:
            status = "finished"
        elif stopped:
            status = "stopped"
        else:
            status = "failed"
        task['status'] = status

        # add run files to the task
        if task['model'] == 'YoloV5' or task['model'] == 'YoloV8':
            yolo_run_folder = self.yolo_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    def inference(self, task_id, task_data):
        """
        Handle the inference task.

        Parameters:
        task_id (int): The ID of the task.
        task_data (dict): Data associated with the task.

        Returns:
        dict: The task with its updated status.
        """
        args = []
        venv_python = None
        run_path = None

        if task_data['model'] == 'YoloV5':
            for key, value in task_data['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')    

            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/yolov5/venv/bin/python'
            run_path = Path(f"data/inference/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        if task_data['model'] == 'YoloV8':
            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/ultralytics/venv/bin/python'
            run_path = Path(f"data/inference/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
            
        if task_data['model'] == 'YoloV5':
            process_str = [venv_python, '-u', 'nets/yolov5/detect.py', *args, *extra_args]
        elif task_data['model'] == 'YoloV8':
            save_dir = Path(__file__).resolve().parent / "nets/ultralytics/runs"
            script = f"""
from ultralytics import YOLO
model = YOLO('{task_data["args"]["weights"]}')
results = model.predict(source="{task_data["args"]["source"]}", conf={task_data["args"]["conf-thres"]}, save=True, project='{save_dir}')
print(results)
"""
            process_str = [venv_python, '-u', '-c', script]
        else:
            raise ValueError(f"Model {task_data['model']} not found")

        cmd_file = run_path / 'meta.log'
        with open(cmd_file, 'w') as f:
            f.write('Command:\n')
            f.write(' '.join(process_str)+'\n\n')
            for key, value in task_data['args'].items():
                f.write(f'{key}: {value}\n')
            if task_data.get('remark') is not None:
                f.write(f"Remark: {task_data['remark']}\n")

        task = {
                        'task_id': task_id,
                        'type': 'inference',
                        'status': "running",
                        'data': str(run_path.resolve()),
                        'command': ' '.join(process_str),
                        'model': task_data['model'],
                    }
        self.current_task = task

        process = subprocess.Popen(process_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_thread = threading.Thread(target=self.log_handler, args=("inference", process, log_file))
        log_thread.start()
        stopped = False
        while process.poll() is None:
            time.sleep(1)
            if self.stop_current_task.is_set():
                self.stop_current_task.clear()
                process.kill()
                stopped = True
                break

        log_thread.join()  # Ensure the logging thread has finished

        if process.poll() == 0:
            status = "finished"
        elif stopped:
            status = "stopped"
        else:
            status = "failed"
        task['status'] = status

        # add run files to the task
        if task['model'] == 'YoloV5' or task['model'] == 'YoloV8':
            yolo_run_folder = self.yolo_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    # export is realtime
    def export(self, task_id, task_data):
        """
        Handle the export task.

        Parameters:
        task_id (int): The ID of the task.
        task_data (dict): Data associated with the task.

        Returns:
        dict: The task with its updated status.
        """
        args = []
        venv_python = None
        run_path = None

        if task_data['model'] == 'YoloV5':
            for key, value in task_data['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')    

            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/yolov5/venv/bin/python'
            run_path = Path(f"data/export/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        if task_data['model'] == 'YoloV8':
            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/ultralytics/venv/bin/python'
            run_path = Path(f"data/export/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()

            
        if task_data['model'] == 'YoloV5':
            process_str = [venv_python, '-u', 'nets/yolov5/export.py', *args, *extra_args]
        elif task_data['model'] == 'YoloV8':
            save_dir = Path(__file__).resolve().parent / "nets/ultralytics/runs"
            script = f"""
from ultralytics import YOLO
model = YOLO('{task_data["args"]["weights"]}')
model.export(format='onnx')
"""
            process_str = [venv_python, '-u', '-c', script]
        else:
            raise ValueError(f"Model {task_data['model']} not found")

        cmd_file = run_path / 'meta.log'
        with open(cmd_file, 'w') as f:
            f.write('Command:\n')
            f.write(' '.join(process_str)+'\n\n')
            for key, value in task_data['args'].items():
                f.write(f'{key}: {value}\n')
            if task_data.get('remark') is not None:
                f.write(f"Remark: {task_data['remark']}\n")

        task = {
                        'task_id': task_id,
                        'type': 'export',
                        'status': "running",
                        'data': str(run_path.resolve()),
                        'command': ' '.join(process_str),
                        'model': task_data['model'],
                    }
        self.current_task = task

        process = subprocess.Popen(process_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_thread = threading.Thread(target=self.log_handler, args=("export", process, log_file))
        log_thread.start()
        stopped = False
        while process.poll() is None:
            time.sleep(1)
            if self.stop_current_task.is_set():
                self.stop_current_task.clear()
                process.kill()
                stopped = True
                break

        log_thread.join()  # Ensure the logging thread has finished

        if process.poll() == 0:
            status = "finished"
        elif stopped:
            status = "stopped"
        else:
            status = "failed"
        task['status'] = status

        # append status to the meta.log file
        with open(cmd_file, 'a') as f:
            f.write(f"Status: {status}\n")

        # get location of exported model
        model_path = Path(task_data['args']['weights'].replace('.pt', '.onnx'))
        model_name = task_data.get('model_name', model_path.name)
        if model_name == ".pt":
            model_name = model_path.name
        uploaded = False
        if model_path.exists():
            uploaded = model_path.parent.name == "uploaded"

        # add run files to the task
        # if task['model'] == 'YoloV5':
        #     yolo_run_folder = self.yolov5_runs_map.get(log_file)
        #     if yolo_run_folder is not None and not uploaded:
        #         yolo_run_folder = str(Path(yolo_run_folder).resolve())
        #         task['artifacts'] = yolo_run_folder
        #         with open(cmd_file, 'a') as f:
        #             f.write(f'artifacts: {yolo_run_folder}\n')
        #         os.symlink(yolo_run_folder, run_path /"run")
        # copy the exported model to the run folder
        if model_path.exists():
            os.rename(model_path, run_path / model_name)
        # # zip the run folder
        # os.chdir(run_path.parent)
        # subprocess.run(['zip', '-r', f'{model_name}.zip', run_path.name])
        # # put the zip file in the run folder
        # os.rename(run_path.parent / f'{model_name}.zip', run_path / f'{model_name}.zip')
        return task 

    def get_current_task(self):
        """
        Get the currently running task.

        Returns:
        dict: The current task.
        """
        return self.current_task
    
    def get_current_task_info(self):
        """
        Get detailed information about the current task.

        Returns:
        dict: Detailed information of the current task.
        """
        if self.current_task is None:
            return None
        else:
            task = self.get_current_task()
            # read logs and send the last 10 lines
            with open(task['log_file'], 'r') as f:
                lines = f.readlines()
                lines = lines[-10:]
                task['latest_log'] = ''.join(lines)
            return task
    
    def get_queued_tasks(self):
        """
        Get a list of tasks in the queue.

        Returns:
        list: A list of queued tasks.
        """
        queued_tasks = []
        for task in self.task_queue.queue:
            task_dict = {}
            task_dict['task_id'] = task[0]
            task_dict['type'] = task[1]
            task_dict['model'] = task[2]['model']

            args = []
            for key, value in task[2]['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')

            extra_args = []
            if task[2].get('extra_args') is not None:
                extra_args = task[2]['extra_args'].split()
            task_dict['args'] = " ".join([*args, *extra_args])
            if task[2].get('remark') is not None:
                task_dict['remark'] = task[2]['remark']

            queued_tasks.append(task_dict)
        return queued_tasks

    def get_tasks(self):
        """
        Get all tasks including current, finished, and queued tasks.

        Returns:
        dict: A dictionary containing all task information.
        """
        return {'current_task': self.get_current_task(),
                 'finished_tasks': self.finished_tasks,
                   'queue': self.get_queued_tasks()}

    def add_task(self, task_type, task_data):
        """
        Add a task to the queue.

        Parameters:
        task_type (str): The type of the task (e.g., 'train', 'validate').
        task_data (dict): Data associated with the task.

        Returns:
        dict: Status message with error or success information.
        """
        task_id = self.task_max_id

        if task_type not in ["train", "validate", "inference", "export"]:
            return {"error": "Task type not found"}, 400
        
        if task_data.get("model") not in  ["YoloV5", "YoloV8"]:
            return {"error": "Model not found"}, 400  # TODO: add YoloV8
        
        if task_data.get("args") is None:
            return {"error": "No arguments provided"}, 400
        
        add_to_queue = self.task_queue.qsize() > 0

        self.task_queue.put((task_id, task_type, task_data))
        self.task_max_id += 1

        if add_to_queue:
            return {"status": f"{task_type} task {task_id} added to queue"}, 200
        else:
            return {"status": f"{task_type} task {task_id} started"}, 201

