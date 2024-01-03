from pathlib import Path
import queue
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
        self.runner = Thread(target=self.train)

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

    def train(self):
        while not self.stop_event.is_set():
            try:
                task_id, task_data = self.task_queue.get(timeout=1)  # Wait for a task
                # Here, start the actual training task

                args = []
                for key, value in task_data['args'].items():
                    args.append(f'--{key}')
                    args.append(f'{value}')

                os.chdir(Path(__file__).resolve().parent)
                venv_python = 'nets/yolov5/venv/bin/python'
                log_file = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log' 
                extra_args = []
                if task_data['extra_args']:
                    extra_args = task_data['extra_args'].split()
                process_str = [venv_python, '-u', 'nets/yolov5/train.py', *args, *extra_args]
                # create a log file
                with open(log_file, 'w') as f:

                    task = {
                        'task_id': task_id,
                        'status': "running",
                        'log_file': log_file,
                        'command': ' '.join(process_str),
                        'model': task_data['model'],
                    }
                    self.current_task = task

                    process = subprocess.Popen(process_str,stdout=f, stderr=f)
                    stopped = False
                    while process.poll() is None:
                        time.sleep(1)
                        if self.stop_current_task.is_set():
                            self.stop_current_task.clear()
                            process.kill()
                            stopped = True
                            break

                    if process.poll() == 0:
                        status = "finished"
                    elif stopped:
                        status = "stopped"
                    else:
                        status = "failed"
                    task['status'] = status

                # move on to the next task
                self.task_queue.task_done()
                self.finished_tasks.append(task)
                self.current_task = None
                
            except queue.Empty:
                continue  # No task, the thread goes back to sleep


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

    def add_task(self, task_data):
        task_id = self.task_max_id
        if task_data.get("model") not in  ["YoloV5", "YoloV8"]:
            return {"error": "Model not found"}, 400  # TODO: add YoloV8
        
        if task_data.get("args") is None:
            return {"error": "No arguments provided"}, 400
        
        add_to_queue = self.task_queue.qsize() > 0

        self.task_queue.put((task_id, task_data))
        self.task_max_id += 1

        if add_to_queue:
            return {"status": "training task added to queue"}, 200
        else:
            return {"status": "training task started"}, 201


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
@app.route('/start-training', methods=['POST'])
def start_training():
    task_data = request.get_json()
    result = manager.add_task(task_data)
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
    
#remove a task from the queue
@app.route('/remove-task/<int:task_id>', methods=['DELETE'])
def remove_from_queue(task_id):
    if task_id == manager.current_task['task_id']:
        manager.stop_task()
    else:
        manager.remove_task_from_queue(task_id)
    return jsonify({"status": "task removed from queue"}), 200

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
