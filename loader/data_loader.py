from torch.utils.data import Dataset, DataLoader
from PIL import Image
from itertools import chain 
from glob import glob
import pandas as pd 
import cv2
import numpy as np
from config import config

class CCDataset(Dataset):
    def __init__(self, file,transform,label=None,is_train = True):
        self.imgs = file
        self.target = label
        self.is_train = is_train
        self.preprocess = transform

    def __getitem__(self,index):
        filename = self.imgs[index] 
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.preprocess(img)
        label = self.target[index] if self.is_train else None
        return img,label
    def __len__(self):
        return len(self.imgs)


def get_classified_files(root,mode="train"):
    if mode != "test": 
        #for train and val       
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading "+mode+" dataset")
        for file in all_images:
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        return np.asarray(all_data_path),np.asarray(labels)
    else:
        print("check the mode please!")

def get_csv_files(root, mode = "train"):
    if mode != "test": 
        #for train and val
        print("loading "+mode+" dataset")
        train_csv = pd.read_csv(config.root+'train.csv')
        train_file = train_csv.id.values
        train_file =  [file+".png" for file in train_file]
        train_file = [config.train_data+file for file in train_file ]
        train_label = train_csv.attribute_ids.values
        train_label = [labels.split(" ") for labels in train_label]
        train_label = [[int(label) for label in labels] for labels in train_label]
        print("get csv file done")
        return np.asarray(train_file),np.asarray(train_label)
    else:
        print("check the mode please!")
        