import tensorflow_datasets as tfds

from tqdm import tqdm
import json
import os

def count_label_appearances_in_split(split):
    dataset, info = tfds.load("coco", split = split, shuffle_files = True, with_info= True)

    labels = info.features["objects"]["label"].names
    amt_of_labels = len(labels)

    label_ids = [i for i in range(amt_of_labels)]

    _assert_all_labels_have_boxes(dataset, split)

    label_appearances = _count_label_appearances(dataset, label_ids, split)

    # Store the amount of boxes there are in each label as a json object
    _save_label_appearances(label_appearances, labels, split)

    

def _save_label_appearances(label_appearances, labels, split):
    named_appearances = {labels[index]: amount for (index, amount) in label_appearances.items()}

    file_name = "counts_for_" + split + ".json"
    path = os.path.join("coco_label_stats", file_name)
    with open(path, "w") as file:
        json.dump(named_appearances, file, indent = 4)

def _assert_all_labels_have_boxes(dataset, split):
    for example in tqdm(dataset, desc = "asserting that the " + split + " split has boxes for every label"):
        example_labels = example["objects"]["label"].numpy()
        boxes = example["objects"]["bbox"].numpy()
        example_id = example["image/id"].numpy()

        if not len(example_labels) == len(boxes):
            raise Exception("Image " + str(example_id) + " contained labels without boxes")
        
def _count_label_appearances(dataset, label_ids, split):
    label_counts = {label: 0 for label in label_ids}

    # Count how many appearances there are for each label
    for example in tqdm(dataset, desc = "Counting the label appearances in the " + split + " split"):
        example_labels = example["objects"]["label"].numpy()

        for label_id in example_labels:
            label_counts[label_id] += 1

    return label_counts

count_label_appearances_in_split("train")
count_label_appearances_in_split("validation")