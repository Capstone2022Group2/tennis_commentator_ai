# not currently working, might need to look at it later

import darwin.importer as importer
from darwin.client import Client
from darwin.importer import get_importer

client = Client.from_api_key("zFjJhE-._rp_OPA4HI2LjZ99u1NFj9rDSL2WdeK3")
dataset = client.get_remote_dataset(dataset_identifier="capstone2022/labeled_tennis")

annotation_paths = [
  "C:/Users/100707158/Downloads/bboxtest2.v7i.voc/valid/",
  "C:/Users/100707158/Downloads/bboxtest2.v7i.voc/train/",
]

parser = get_importer("pascal_voc")

# annotation_paths = [
#   "C:/Users/100707158/Downloads/valid/_annotations.coco.json",
#   "C:/Users/100707158/Downloads/valid/train/_annotations.coco.json",
# ]

# parser = get_importer("coco")

importer.import_annotations(dataset, parser, annotation_paths, append=True)
