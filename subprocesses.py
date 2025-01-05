import subprocess
import os

def clone_repo(github_url, local_path):
    try:
        if not os.path.exists(local_path):
            subprocess.run(["git", "clone", github_url, local_path])
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print("Git is not installed or not found in PATH.")


clone_repo("https://github.com/ArshanBhanage/ror-assignment.git", "user_projects")