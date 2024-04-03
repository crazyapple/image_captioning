import os
import shutil
import json
import unicodedata
import re

DATASET_FOLDER_PATH = 'dataset'

IMAGE_FOLDER_PATH = 'images'
CAPTION_FOLDER_PATH = 'captions'
INSTANCE_FOLDER_PATH = 'instances'

ANNOTATION_FOLDER_PATH = 'annotations'

# Convert the unicode sequence to ascii
def unicode_to_ascii(s):

  # Normalize the unicode string and remove the non-spacking mark
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

# Preprocess the sequence
def preprocess_sentence(w):

  # Clean the sequence
  w = unicode_to_ascii(w.lower().strip())

  # Create a space between word and the punctuation following it
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  return w

caption_json = os.path.join(DATASET_FOLDER_PATH, ANNOTATION_FOLDER_PATH, 'captions_val2017.json')
instance_json = os.path.join(DATASET_FOLDER_PATH, ANNOTATION_FOLDER_PATH, 'instances_val2017.json')
img_files = os.listdir(os.path.join(DATASET_FOLDER_PATH, IMAGE_FOLDER_PATH))

captions = dict()
captions_file = open(caption_json)
captions_data = json.load(captions_file)
for annotation in captions_data['annotations']:
    image_id = str(annotation['image_id'])
    if captions.get(image_id):
        captions[image_id].append('\n' + preprocess_sentence(annotation['caption']))
    else:
        captions[image_id] = list()
        captions[image_id].append(preprocess_sentence(annotation['caption']))
captions_file.close()

instances = dict()
instances_file = open(instance_json)
instances_data = json.load(instances_file)
for annotation in instances_data['annotations']:
    image_id = str(annotation['image_id'])
    if instances.get(image_id):
        instances[image_id].append('\n' + str(annotation['category_id']) + ' ' + ' '.join(map(str, annotation['bbox'])))
    else:
        instances[image_id] = list()        
        instances[image_id].append(str(annotation['category_id']) + ' ' + ' '.join(map(str, annotation['bbox'])))

f_classes = open(os.path.join(DATASET_FOLDER_PATH, 'classes.txt'), 'w')
for class_info in instances_data['categories']:
    f_classes.write(class_info['name'] + '\n')

instances_file.close()

for img_file in img_files:
    img_id = str(int(img_file[:-4]))
    if instances.get(img_id) is not None and len(instances[img_id]) >= 5:
        f_caption = open(os.path.join(DATASET_FOLDER_PATH, CAPTION_FOLDER_PATH, img_file[:-3] + 'txt'), 'w')
        f_caption.writelines(captions[img_id])

        f_instance = open(os.path.join(DATASET_FOLDER_PATH, INSTANCE_FOLDER_PATH, img_file[:-3] + 'txt'), 'w')
        f_instance.writelines(instances[img_id])
    else:
        shutil.move(os.path.join(DATASET_FOLDER_PATH, IMAGE_FOLDER_PATH, img_file), os.path.join('out_image', img_file))
