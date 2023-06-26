from utils import coco_utils

vehicles = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

coco_utils.process_data_from_class_subset(vehicles, "vehicles")