#pascalvocdatasetnesnetespiti

# import deeplake
# ds = deeplake.load('hub://activeloop/pascal-voc-2007-train-val')
# import deeplake
# ds = deeplake.load('hub://activeloop/pascal-voc-2007-test')
# dataloader = ds.tensorflow()


import tensorflow_datasets as tfds

# Pascal VOC 2007 veri setini indirin
dataset, info = tfds.load("voc/2007", with_info=True)

import tensorflow_hub as hub

# SSD MobilNet V2 modeli
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))  # SSD modelinin girdi boyutu
    img = np.expand_dims(img, axis=0)
    return img

def detect_objects(image_path, model):
    img = load_image(image_path)
    result = model(img)
    return result

def display_detections(image_path, detections, threshold=0.5):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape
    
    for i in range(len(detections['detection_scores'][0])):
        if detections['detection_scores'][0][i].numpy() > threshold:
            box = detections['detection_boxes'][0][i].numpy()
            y_min, x_min, y_max, x_max = box
            y_min, y_max = int(y_min * height), int(y_max * height)
            x_min, x_max = int(x_min * width), int(x_max * width)
            
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            label = detections['detection_classes'][0][i].numpy()
            score = detections['detection_scores'][0][i].numpy()
            plt.text(x_min, y_min, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

# Örnek görüntü yolu
image_path = 'C:\\Users\\Hazar\\4.jpeg'

# Nesne tespiti yap
detections = detect_objects(image_path, model)

# Sonuçları görüntüle
display_detections(image_path, detections)
