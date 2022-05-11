
#!/bin/usr/python
# author Thibault Tabarin
import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import json, random, os, sys, yaml, pathlib, cv2
import numpy as np
import pandas as pd
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, Metadata, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import warnings
warnings.filterwarnings("ignore")

# This classes_dict be insert into
classes_dict = {'pallette_square': 0, 'top_left': 1,
    'top_right': 2, 'bot_left': 3, 'bot_right': 4}

def setup_model(processor='cpu'):
    '''
    Construct the model based on pretrained model COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3
    and fine tune model model_final.pth, model_final.pth should in ./output/model_final.pth
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.DEVICE = processor
    predictor = DefaultPredictor(cfg)
    return predictor

def import_image(input_file):
    ''' import the image to np array format'''

    return cv2.imread(input_file)

def predict(im, predictor):
    ''' just predict the instance segementation
    the function output is an detectron2 instances object'''
    output = predictor(im)
    instances = output['instances']
    return  instances

def create_metadata(classes_dict):
    '''
    Create an object metadata to create prediction image (visualization)
    Check detctron2 documentation for more information
    '''
    thing_classes = list(classes_dict.keys())
    thing_dataset_= {i+1: i for i in range(len(classes_dict))}
    metadata = Metadata(evaluator_type='coco', image_root='.',
                        json_file='', name='metadata',
                        thing_classes=thing_classes,
                        thing_dataset_id_to_contiguous_id=thing_dataset_)
    return metadata


def prediction_image(image_arr, instances, classes_dict):
    '''
    Create a prediction image for visualization
    Check detctron2 documentation for more information
    '''
    metadata = create_metadata(classes_dict)
    v = Visualizer(image_arr[:, :, ::-1], metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels. This option is only available for segmentation models

    out = v.draw_instance_predictions(instances.to("cpu"))
    pred_image_arr = out.get_image()[:, :, ::-1]

    return pred_image_arr

def get_class(instances, classes_dict, class_name):

    '''
    Get the instance corresponding to a specific class object (class_name: from the classes_dict)
    class object : 'pallette_square', 'top_left': 1, 'top_right', 'bot_left', 'bot_right'}

    '''
    instance_index = classes_dict[class_name]
    class_object = instances[instances.pred_classes == instance_index]

    return class_object

def get_centers(class_object):

    '''
    Create a dictionnary with
    center x,y, number of instance for the class_object
    If there are several instance for class, we take the instance of the higest score
    If the class does exist (len(instance)=0) return empty value
    '''

    dict_={'x':None,'y':None, 'number':None}
    if len(class_object) >=1:

        index_max = np.argmax(class_object.scores.cpu().numpy()).tolist()
        x, y = class_object.pred_boxes.get_centers().cpu().numpy()[index_max].tolist()
        dict_= {'x':round(x),'y':round(y),'number':len(class_object)}
    return dict_

def get_palette_metadata(classes_dict, instances):

    '''
    Create a dictionnary key : 'top_left', 'top_right'....
    value {'x':123,'y':456, 'number':1} x, y center coordinate
    '''

    square_dict = {'top_left': 1, 'top_right': 2,
                   'bot_left': 3, 'bot_right': 4}

    dict_result = {}
    for index, class_name in enumerate(square_dict):

        class_object = get_class(instances, classes_dict, class_name)
        dict_result[class_name] = get_centers(class_object)

    return dict_result

def main(input_file, output_csv, output_json=None, output_pred=None, classes_dict=classes_dict, processor='cpu'):

    predictor = setup_model(processor=processor)
    image = import_image(input_file)
    instances = predict(image, predictor)
    dict_result = get_palette_metadata(classes_dict, instances)

    if output_json != None:
        with open(output_json, 'w') as f:
                json.dump(dict_result, f)


    df = pd.DataFrame.from_dict(dict_result).T
    df.to_csv(output_csv, index=True)

    # Save prediction image if output_pred not None
    if output_pred != None:
        pred_image_arr = prediction_image(image, instances,classes_dict=classes_dict)
        cv2.imwrite(output_pred, np.array(pred_image_arr, dtype=np.uint8))
        #cv2.imwrite(output_pred, pred_image_arr)

    return dict_result

if __name__ == '__main__':

    '''
    Usage : 2 usages
    1- metadata.py <input_file> <output_csv>
    2- metadata.py <input_file> <output_csv> <output_json> <output_pred>
    '''
    if len(sys.argv) == 3: # case with 2 arguments input
        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        main(input_file, output_csv)

    elif len(sys.argv) == 4:

        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        output_json = sys.argv[3]
        main(input_file, output_csv, output_json = output_json)

    elif len(sys.argv) == 5:

        input_file = sys.argv[1]
        output_csv = sys.argv[2]
        output_json = sys.argv[3]
        output_pred = sys.argv[4]
        main(input_file, output_csv, output_json = output_json, output_pred=output_pred)


    else:
        print ("Wrong number of inputs! Check the documentation")
