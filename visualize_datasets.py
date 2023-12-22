import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming tasks_df and projects_df are already created as per the previous script
# Replace the file paths with your actual file paths
projects_csv_file = 'cvat_projects.csv'
tasks_csv_file = 'cvat_tasks.csv'

# Load the data into pandas DataFrames
projects_df = pd.read_csv(projects_csv_file)
tasks_df = pd.read_csv(tasks_csv_file)

# Histogram of Task Status
plt.figure(figsize=(10, 6))
sns.countplot(x='status', data=tasks_df)
plt.title('Distribution of Task Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Bar Plot of Tasks per Project
task_count_per_project = tasks_df['project_name'].value_counts()
plt.figure(figsize=(10, 6))
task_count_per_project.plot(kind='bar')
plt.title('Number of Tasks per Project')
plt.xlabel('Project Name')
plt.ylabel('Number of Tasks')
plt.xticks(rotation=45)
plt.show()

# Time Series of Task Creation
tasks_df['created_date'] = pd.to_datetime(tasks_df['created_date'])
tasks_df.set_index('created_date', inplace=True)
tasks_per_day = tasks_df.resample('D').size()
plt.figure(figsize=(12, 6))
tasks_per_day.plot()
plt.title('Tasks Creation Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tasks Created')
plt.show()
