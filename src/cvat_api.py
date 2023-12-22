import base64
import io
import time
import requests
import json
from tqdm import tqdm
import PIL
import numpy as np
"""
This module contains the CVATAPI class, which is used to interact with the CVAT API.
"""


class CVATAPI:
    def __init__(self, cvat_url):
        self.cvat_url = cvat_url
        self.session = None
    
    def authenticate(self, username, password):
        """
        Authenticate to CVAT.

        Args:
        username (str): The username of the CVAT user.
        password (str): The password of the CVAT user.

        """
        session = requests.Session()
        headers = {'Content-Type': 'application/json'}
        auth_response = session.post(f"{self.cvat_url}/api/auth/login", 
                                    data=json.dumps({'username': username, 'password': password}),
                                    headers=headers)
        auth_response.raise_for_status()
        self.session = session 


    def is_logged_in(self):
        """
        Check if the user is logged in to CVAT.
        """
        if self.session is None:
            return False
        
        response = self.session.get(f"{self.cvat_url}/api/users/self")
        if response.status_code == 200:
            return True
        elif response.status_code == 403:
            return False
        else:
            response.raise_for_status()

    def logout(self):
        """
        Logout from CVAT.
        """
        if not self.is_logged_in():
            return 
        
        response = self.session.post(f"{self.cvat_url}/api/auth/logout")
        response.raise_for_status()
        self.session = None

    def fetch_all_pages(self, url):
        """Fetch all pages from a paginated API."""

        results = []
        while url:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            results.extend(data['results'])
            url = data.get('next')
        return results


    def fetch_page(self, url_with_page):
        """Fetch a single page from a paginated API."""
        response = self.session.get(url_with_page)
        response.raise_for_status()
        return response.json()


    def fetch_specific_page(self, url, page_number):
        """Fetch a specific page from a paginated API."""
        response = self.session.get(url, params={'page': page_number})
        response.raise_for_status()
        return response.json()
    
    def get_task_info(self, task_id):
        """
        Get information about a task.

        Args:
        task_id (int): The ID of the task to get information for.

        Returns:
        dict: The task information.
        """
        response = self.session.get(f"{self.cvat_url}/api/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_task_preview(self, task_id):
        """
        Get a preview image of a task.

        Args:
        task_id (int): The ID of the task to get a preview for.

        """
        try:
            response = self.session.get(f"{self.cvat_url}/api/tasks/{task_id}/preview", headers={'Content-Type': 'image/jpeg'})
            response.raise_for_status()
            bytes = response.content

            return f"data:image/png;base64,{base64.b64encode(bytes).decode()}"
        except Exception as e:
            print(e)
            return None
        
    def get_project_preview(self, project_id):
        """
        Get a preview image of a project.

        Args:
        project_id (int): The ID of the project to get a preview for.

        """
        try:
            response = self.session.get(f"{self.cvat_url}/api/projects/{project_id}/preview", headers={'Content-Type': 'image/jpeg'})
            response.raise_for_status()
            bytes = response.content

            return f"data:image/png;base64,{base64.b64encode(bytes).decode()}"
        except Exception as e:
            print(e)
            return None


    def _request_download_annotation(self, task_id, file_format='YOLO 1.1', save_images=False):
        """
        Request server to prepare the annotation dump.

        Args:
        task_id (int): The ID of the task to request annotations for.
        file_format (str): The format of the annotation file (default: 'YOLO 1.1').
        save_images (bool): Whether to also save the images (default: False).

        Returns:
        bool: success status.
        """
        endpoint = 'dataset' if save_images else 'annotations'

        dump_url = f"{self.cvat_url}/api/tasks/{task_id}/{endpoint}?format={file_format}"
        response = self.session.get(dump_url)
        
        if response.status_code == 202:
            print(f"Dump of annotations from task {task_id} has been started. Awaiting completion...")
            return False
        elif response.status_code == 201:
            print(f"Annotations file from task {task_id} is ready to download.")
            return True
        else:
            response.raise_for_status()


    def download_annotation(self, task_id, file_format, save_path, save_images=False):
        """
        Download annotations for a given task ID from CVAT.

        Args:
        task_id (int): The ID of the task to download annotations for.
        file_format (str): The format of the annotation file (e.g., 'CVAT XML 1.1 for images').
        save_path (str): The path where the annotation file will be saved.
        save_images (bool): Whether to also save the images (default: False).
        """
        while not self._request_download_annotation(task_id, file_format, save_images=save_images):
            time.sleep(1)

        endpoint = 'dataset' if save_images else 'annotations'

        # The URL to download the annotation
        annotation_url = f"{self.cvat_url}/api/tasks/{task_id}/{endpoint}?format={file_format}&action=download"
        headers = {'Content-Type': 'application/json'}

        # Initial request to get the content-length header for total size
        with self.session.get(annotation_url, headers=headers, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192  # size of each chunk

            # Setup the progress bar
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading annotations for task {task_id}")

            # Download with progress bar
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("[ERROR] Transferred file is corrupted")
            exit(1)

        print(f"Annotations for task {task_id} downloaded successfully.")


