"""
AI based waste detection-Eco Guardian
Mainly used roboflow for dataset.
"""

#If you are using roboflow, then install otherwise remove the few lines.
!pip install -q ultralytics roboflow 

from roboflow import Roboflow
rf = Roboflow(api_key="ROBOFLOW API") #Remove the "ROBOFLOW API" and paste your api here
project = rf.workspace("walkman").project("taco-trash-annotations-in-context-s6e6j") # change the project details also
version = project.version(1)
dataset = version.download("yolov8")

"""
Here few lines to check counts dataset, to check whether dataset is balance or not
"""
!cat /content/TACO:-Trash-Annotations-in-Context-Dataset-1/data.yaml

# file counts
!echo "Train images: $(ls /content/TACO:-Trash-Annotations-in-Context-Dataset-1/train/images | wc -l)"
!echo "Train labels: $(ls /content/TACO:-Trash-Annotations-in-Context-Dataset-1/train/labels | wc -l)"

#Importing the required packages
import os
import yaml
import shutil
from ultralytics import YOLO
from google.colab import files

#Checking classes first
def show_classes(dataset_path):
    with open(f'{dataset_path}/data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    print(f"Total classes: {data['nc']}\n")
    for i, name in enumerate(data['names']):
        print(f"{i}: {name}")

    return data['names']

show_classes('/content/TACO:-Trash-Annotations-in-Context-Dataset-1')

"""
From here ownwards meraging the 56 class to 5 class to remove the imbalance of the datasets.
"""

def simple_merge(dataset_path, output_name="merged"):

    with open(f'{dataset_path}/data.yaml', 'r') as f:
        data = yaml.safe_load(f)

    old_names = data['names']
    print(f" Original: {len(old_names)} classes\n")

    mapping = {}
    for i, name in enumerate(old_names):
        n = name.lower()

        if any(word in n for word in ['plastic', 'styrofoam', 'poly']):
            mapping[i] = 0  # plastic
            print(f"  {i}: {name} → plastic")
        elif any(word in n for word in ['metal', 'aluminium', 'aluminum', 'tin', 'can', 'foil']):
            mapping[i] = 1  # metal
            print(f"  {i}: {name} → metal")
        elif 'glass' in n:
            mapping[i] = 2  # glass
            print(f"  {i}: {name} → glass")
        elif any(word in n for word in ['paper', 'cardboard', 'carton']):
            mapping[i] = 3  # paper
            print(f"  {i}: {name} → paper")
        else:
            mapping[i] = 4  # organic/other
            print(f"  {i}: {name} → organic")

    print("\n Starting merge process...\n")

    new_path = f"/content/{output_name}"
    os.makedirs(new_path, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        img_src = f"{dataset_path}/{split}/images"
        lbl_src = f"{dataset_path}/{split}/labels"

        if not os.path.exists(img_src):
            print(f"  Skipping {split} (not found)")
            continue

        img_dst = f"{new_path}/{split}/images"
        lbl_dst = f"{new_path}/{split}/labels"
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        img_count = 0
        for img in os.listdir(img_src):
            shutil.copy2(f"{img_src}/{img}", f"{img_dst}/{img}")
            img_count += 1

        lbl_count = 0
        for lbl in os.listdir(lbl_src):
            with open(f"{lbl_src}/{lbl}", 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    old_id = int(parts[0])
                    new_id = mapping[old_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")

            with open(f"{lbl_dst}/{lbl}", 'w') as f:
                f.writelines(new_lines)
            lbl_count += 1

        print(f" {split}: {img_count} images, {lbl_count} labels")

    new_yaml = {
        'path': new_path, #Creating new file paths and news categories
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 5,
        'names': ['plastic', 'metal', 'glass', 'paper', 'organic']
    }

    with open(f"{new_path}/data.yaml", 'w') as f:
        yaml.dump(new_yaml, f)

    print(f"\n SUCCESS! Merged dataset created!")
    print(f" Location: {new_path}")
    print(f" New classes (5): plastic, metal, glass, paper, organic\n")

    return new_path

new_dataset = simple_merge(
    '/content/TACO:-Trash-Annotations-in-Context-Dataset-1',
    'taco_5classes'
)

print(" Verifying new data.yaml:")
with open(f'{new_dataset}/data.yaml', 'r') as f:
    merged_data = yaml.safe_load(f)
    print(f"Classes: {merged_data['nc']}")
    print(f"Names: {merged_data['names']}")

print("\n Ready to train! Use this path:")
print(f"data='/content/taco_5classes/data.yaml'")

new_dataset = simple_merge(
    '/content/TACO:-Trash-Annotations-in-Context-Dataset-1',
    'taco_5classes'
)

print("🔍 Verifying new data.yaml:")
with open(f'{new_dataset}/data.yaml', 'r') as f:
    merged_data = yaml.safe_load(f)
    print(f"Classes: {merged_data['nc']}")
    print(f"Names: {merged_data['names']}")


"""
Model training starts here ownwards..........
"""
model = YOLO('yolov8n.pt')

model.train(
    data='/content/taco_5classes/data.yaml',
    epochs = 100,
    imgsz=640,
    batch = 16,  
    patience = 50,
    plots = True,
    project = 'runs/train',
    device = 0,
    name = 'taco'
)

print(" Training Results:")
print("="*50)


best_model = YOLO('/content/runs/detect/runs/train/taco/weights/best.pt')

metrics = best_model.val()

print(f"\n Final Performance:")
print(f"mAP@50:    {metrics.box.map50:.3f} ({metrics.box.map50*100:.1f}%)")
print(f"mAP@50-95: {metrics.box.map:.3f} ({metrics.box.map*100:.1f}%)")
print(f"Precision: {metrics.box.mp:.3f} ({metrics.box.mp*100:.1f}%)")
print(f"Recall:    {metrics.box.mr:.3f} ({metrics.box.mr*100:.1f}%)")


print("Model downloades and results")
print("="*50)

files.download('/content/runs/detect/runs/train/taco/weights/best.pt')
files.download('/content/runs/detect/runs/train/taco/results.png')
files.download('/content/runs/detect/runs/train/taco/confusion_matrix.png')

print(" Model and results downloaded!")
