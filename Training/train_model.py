# import standart libraries
import json, random, os, sys, yaml, pathlib, cv2
import numpy as np

# import detectron2 related
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, Metadata, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def import_config(config_data):

    config_file = json.load(open(config_data))

    #ENHANCE = config_file['ENHANCE']
    annotation_prefix = config_file['annotation_prefix']
    dataset_json = config_file['dataset_json']
    data_prefix = config_file['data_prefix']
    labels_file = config_file['labels']
    labels_f = open(labels_file, 'r')
    data = labels_f.read()
    labels_list = [l for l in data.split("\n") if l != '']

    return annotation_prefix, dataset_json, data_prefix, labels_list  # , ENHANCE


def register_data(config_data):

    annotation_prefix,\
        dataset_json, prefix,\
        labels_list = import_config(config_data)

    mapping = {i+1: i for i in range(len(labels_list))}
    conf = json.load(open(dataset_json))
    metadata = None  # Need it in outer block for reuse
    train = []

    for img_dir in conf.keys():
        #ims = f'{prefix}{img_dir}'
        ims = os.path.join(prefix, img_dir)
        for dataset in conf[img_dir]:
            json_file = os.path.join(annotation_prefix, dataset)
            #json_file = f'datasets/{dataset}'
            name = dataset.split('.')[0]
            train.append(name)
            register_coco_instances(name, {}, json_file, ims)

    return train


def Train(train, nb_classes):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = tuple(train)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nb_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

def main(config_data):

    # Using DatasetCatalog get the number of class
    #
    nb_classes = 5
    train = register_data(config_data)
    Train(train, nb_classes)

if __name__ == '__main__':

    config_data = sys.argv[1]
    main(config_data)
