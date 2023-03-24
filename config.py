import os

DATA_PATH_MINI = '../data/miniimagenet'
EPSILON = 1e-8
if DATA_PATH_MINI is None:
    raise Exception('Configure your data folder location in config.py before continuing!')

base = '../data'
ISIC_path = os.path.join(base, 'ISIC')
ChestX_path = os.path.join(base, 'ChestX_images_labels')
CropDisease_path = os.path.join(base, 'CropDiseases')
EuroSAT_path = os.path.join(base, 'EuroSAT')


