from pathlib import Path
import queue
import threading
import time
from flask import Flask, jsonify, request
import subprocess
from threading import Thread, Event
import os
import datetime

class TaskManager:
    def __init__(self):
        self.task_max_id = 1
        self.finished_tasks = []
        self.current_task = None
        self.task_queue = queue.Queue()
        self.stop_event = Event()
        self.stop_current_task = Event()
        self.runner = Thread(target=self.run)
        self.yolov5_runs_map = {}

    def terminate(self):
        self.stop_event.set()
        self.runner.join()

    def stop_task(self):
        if self.current_task is None:
            return {"error": "No task is running"}, 400
        self.stop_current_task.set()

    def empty_tasks(self):
        self.task_queue = queue.Queue()

    def remove_task_from_queue(self, task_id):
        tasks = []
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task[0] != task_id:
                tasks.append(task)
        for task in tasks:
            self.task_queue.put(task)

    def log_handler(self, task_type, process, log_file_path):
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
                        if self.yolov5_runs_map.get(log_file_path) is None and "Plotting labels to" in output:
                            run_id = output.split("Plotting labels to ")[1].split("/labels.jpg")[0]
                            self.yolov5_runs_map[log_file_path] = run_id

                    if task_type == "validate" or task_type == "inference":
                        # get the run id ("...nets/yolov5/runs/train/exp%run_id%....")
                        if self.yolov5_runs_map.get(log_file_path) is None and "Results saved to [1m" in output:
                            run_id = output.split("Results saved to [1m")[1].split("[0m")[0]
                            self.yolov5_runs_map[log_file_path] = run_id

                    log_file.write(output)
                    log_file.flush()

    def run(self):
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
        args = []
        for key, value in task_data['args'].items():
            args.append(f'--{key}')
            args.append(f'{value}')

        os.chdir(Path(__file__).resolve().parent)
        venv_python = 'nets/yolov5/venv/bin/python'
                    
        run_path = Path(f"data/train/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
        process_str = [venv_python, '-u', 'nets/yolov5/train.py', *args, *extra_args]
                    
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
                f.write(f'Weights: {task_data["weights"]}\n')

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
        if task['model'] == 'YoloV5':
            yolo_run_folder = self.yolov5_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    def validate(self, task_id, task_data):
        args = []
        for key, value in task_data['args'].items():
            args.append(f'--{key}')
            args.append(f'{value}')

        os.chdir(Path(__file__).resolve().parent)
        venv_python = 'nets/yolov5/venv/bin/python'
                    
        run_path = Path(f"data/val/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
        process_str = [venv_python, '-u', 'nets/yolov5/val.py', *args, *extra_args]

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
            f.write(f'Weights: {task_data["weights"]}\n')
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
        if task['model'] == 'YoloV5':
            yolo_run_folder = self.yolov5_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    def inference(self, task_id, task_data):
        args = []
        for key, value in task_data['args'].items():
            args.append(f'--{key}')
            args.append(f'{value}')

        os.chdir(Path(__file__).resolve().parent)
        venv_python = 'nets/yolov5/venv/bin/python'
                    
        run_path = Path(f"data/inference/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
        process_str = [venv_python, '-u', 'nets/yolov5/detect.py', *args, *extra_args]

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
        if task['model'] == 'YoloV5':
            yolo_run_folder = self.yolov5_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 
    
    # export is realtime
    def export(self, task_id, task_data):
        args = []
        for key, value in task_data['args'].items():
            args.append(f'--{key}')
            args.append(f'{value}')

        os.chdir(Path(__file__).resolve().parent)
        venv_python = 'nets/yolov5/venv/bin/python'
                    
        run_path = Path(f"data/export/{task_data['model']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        run_path.mkdir(parents=True, exist_ok=True)
        log_file = run_path / 'stdout.log'

        extra_args = []
        if task_data.get('extra_args') is not None:
            extra_args = task_data['extra_args'].split()
        process_str = [venv_python, '-u', 'nets/yolov5/export.py', *args, *extra_args]

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

        # add run files to the task
        if task['model'] == 'YoloV5':
            yolo_run_folder = self.yolov5_runs_map.get(log_file)
            if yolo_run_folder is not None:
                yolo_run_folder = str(Path(yolo_run_folder).resolve())
                task['artifacts'] = yolo_run_folder
                with open(cmd_file, 'a') as f:
                    f.write(f'artifacts: {yolo_run_folder}\n')
                os.symlink(yolo_run_folder, run_path /"run")
        return task 

    def get_current_task(self):
        return self.current_task
    
    def get_current_task_info(self):
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
        queued_tasks = []
        for task in self.task_queue.queue:
            task_dict = {}
            task_dict['task_id'] = task[0]
            task_dict['model'] = task[1]['model']
            task_dict['status'] = "not started"
            args = []
            for key, value in task[1]['args'].items():
                args.append(f'--{key}')
                args.append(f'{value}')

            os.chdir(Path(__file__).resolve().parent)
            venv_python = 'nets/yolov5/venv/bin/python'
            extra_args = []
            if task[1]['extra_args']:
                extra_args = task[1]['extra_args'].split()
            process_str = [venv_python, '-u', 'nets/yolov5/train.py', *args, *extra_args]
            task_dict['command'] = ' '.join(process_str)
            queued_tasks.append(task_dict)
        return queued_tasks

    def get_tasks(self):
        return {'current_task': self.get_current_task(),
                 'finished_tasks': self.finished_tasks,
                   'queue': self.get_queued_tasks()}

    def add_task(self, task_type, task_data):
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

