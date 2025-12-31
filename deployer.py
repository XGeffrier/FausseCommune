import os
import re
import shutil

DEPLOY_FOLDER = "./to_deploy"

# 1 - Create deploy folder
if os.path.exists(DEPLOY_FOLDER):
    shutil.rmtree(DEPLOY_FOLDER)
os.makedirs(DEPLOY_FOLDER, exist_ok=True)

necessary_files = ["main.py", "requirements.txt", "back/game.py", "back/markov/math_utils.py"]
necessary_folders = ["back/data_interfaces", "templates"]

for file in necessary_files:
    destination_path = os.path.join(DEPLOY_FOLDER, file)
    destination_folder = os.path.dirname(destination_path)
    os.makedirs(destination_folder, exist_ok=True)
    shutil.copy(file, os.path.join(DEPLOY_FOLDER, file))
for folder in necessary_folders:
    shutil.copytree(folder, os.path.join(DEPLOY_FOLDER, folder))

# 2 - Remove comments of HMTL and JS code inside templates/*.html
for file in os.listdir(os.path.join(DEPLOY_FOLDER, "templates")):
    if file.endswith(".html"):
        filepath = os.path.join(DEPLOY_FOLDER, "templates", file)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        no_comment = re.sub(r"\/\*[\s\S]*?\*\/|([^\\:]|^)\/\/.*|<!--[\s\S]*?-->$", "", content, flags=re.MULTILINE)
        no_double_return = re.sub(r'\n\s*\n', '\n', no_comment)
        no_double_space = re.sub(' +', ' ', no_double_return)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(no_double_space)

