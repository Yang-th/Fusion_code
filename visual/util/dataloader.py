from torch.utils.data.dataset import TensorDataset
from PIL import Image
import numpy as np
import torch
import os
cate2label = {0: 'anger',1: 'disgust',2: 'fear',3: 'happiness',4: 'sadness',5: 'surprise',
                  'anger': 0,'disgust': 1,'fear': 2,'happiness': 3,'sadness': 4,'surprise': 5}
class Enterfacedataset(TensorDataset):
    def __init__(self,root_train, list_train):
        self.Root_train = root_train
        self.List_train = list_train
        self.video_name = []
        self.label = []

        with open(self.List_train, 'r') as imf:
            index = []
            for id, line in enumerate(imf):
                video_label = line.strip().split()

                self.video_name.append(video_label[0])  # name of video
                self.label.append(cate2label[video_label[1]])  # label of video



    def __getitem__(self, item):
        path = self.Root_train + '/' + self.video_name[item]
        img = []
        files = os.listdir(path)
        files.sort(key=lambda x: int(x[:-4]))
        for i in range(len(files)):
            faceimg = Image.open(path + '/' + files[i]) # .convert('RGB')
            pil_image =  np.array(faceimg)

            img.append(pil_image)
        label = self.label[item]
        label = torch.from_numpy(np.array(label))
        img = np.array(img)
        # img = np.expand_dims(img, 0)
        img = np.transpose(img,[3,0,1,2])
        img = torch.from_numpy(img)
        return img,label

    def __len__(self):
        return len(self.label)