import os
import torch
import pandas as pd
import nibabel as nb
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

class ProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.images = []
        self.labels = []
        self.image_paths = []
        self.frame_num = []
        
        model_mapping_dict = {'TrioTim':0, 'Prisma_fit':1, 'Verio':2, 'Biograph_mMR':3}
        
#         cnt = 0
        for idx, row in tqdm(data_info.iterrows()):
            img = nb.load(row['preprocessed_3_path']).get_data()
            if img.shape[0] == 256: ### to avoid biasing the model with different view of the image acuisition
                continue
                
            img = self.intensity_normalization(img)    
            img = self.image_padding(img)
            
            for i in range(img.shape[2]):
                if np.sum(img[:, :, i]) == 0:
                    continue
                self.images.append(img[:, :, i])
                self.labels.append(model_mapping_dict[row['Mfg Model']])
                self.image_paths.append(row['preprocessed_3_path'])
                self.frame_num.append(i)
#             cnt += 1
#             if cnt >= 5:
#                 break
#             del img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx]))
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_paths[idx], 'frame_num': self.frame_num[idx]}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)
    
    def image_padding(self, raw):
        shape = raw.shape
        marginal = np.zeros((256, 256, 256))
        idx_0 = int((256 - shape[0]) / 2.0)
        idx_1 = int((256 - shape[1]) / 2.0)
        idx_2 = int((256 - shape[2]) / 2.0)

        marginal[idx_0 : idx_0 + shape[0], idx_1: idx_1 + shape[1], idx_2: idx_2 + shape[2] ] = raw
        return marginal


class MemoryEfficientProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.image_path_and_frame_num = []
        self.labels = []
        
        model_mapping_dict = {'TrioTim':0, 'Prisma_fit':1, 'Verio':2, 'Biograph_mMR':3}
        
        for idx, row in tqdm(data_info.iterrows()):
            img = nb.load(row['preprocessed_3_path']).get_data()
            if img.shape[0] == 256: ### to avoid biasing the model with different view of the image acuisition
                continue
                
            for i in range(img.shape[2]):
                if np.sum(img[:, :, i]) == 0:
                    continue
                self.image_path_and_frame_num.append((row['preprocessed_3_path'], i))
                self.labels.append(model_mapping_dict[row['Mfg Model']])
            del img

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, frame_num = self.image_path_and_frame_num[idx]
        
        full_brain_img = nb.load(img_path).get_data()
        img = full_brain_img[:, :, frame_num]
        del full_brain_img
        
        img = self.intensity_normalization(img)    
        img = self.image_padding(img)
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': img_path, 'frame_num': frame_num}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)
    
    def image_padding(self, raw):
        shape = raw.shape
        marginal = np.zeros((256, 256))
        idx_0 = int((256 - shape[0]) / 2.0)
        idx_1 = int((256 - shape[1]) / 2.0)

        marginal[idx_0 : idx_0 + shape[0], idx_1: idx_1 + shape[1]] = raw
        return marginal
    
    
class JpegProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        model_mapping_dict = {'TrioTim':0, 'Prisma_fit':1, 'Verio':2, 'Biograph_mMR':3}
        
        for idx, row in tqdm(data_info.iterrows()):
            self.image_path.append(row['preprocessed_3_path_jpeg'])
            self.labels.append(model_mapping_dict[row['Mfg Model']])
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        img = self.intensity_normalization(img)    
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)
    

class JpegBalancedProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        model_mapping_dict = {'TrioTim':0, 'Prisma_fit':1, 'Verio':2, 'Biograph_mMR':3}
        
        images_for_each_label = defaultdict(list)
        
        for idx, row in data_info.iterrows():
            images_for_each_label[model_mapping_dict[row['Mfg Model']]].append((row['preprocessed_3_path_jpeg'], row['frame_num']))
        
        min_count = len(data_info)
        for label in images_for_each_label:
            min_count = min(min_count, len(images_for_each_label[label]))
        
        print(min_count)
        
        for label in images_for_each_label:
            print(label)
            for image_path, frame_num in images_for_each_label[label][:min_count]:
                self.image_path.append(image_path)
                self.labels.append(label)
                self.frame_num.append(frame_num)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        img = self.intensity_normalization(img)    
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)

    
class JpegBalancedProcessedDatasetForResNet(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        model_mapping_dict = {'TrioTim':0, 'Prisma_fit':1, 'Verio':2, 'Biograph_mMR':3}
        
        images_for_each_label = defaultdict(list)
        
        for idx, row in data_info.iterrows():
            images_for_each_label[model_mapping_dict[row['Mfg Model']]].append((row['preprocessed_3_path_jpeg'], row['frame_num']))
        
        min_count = len(data_info)
        for label in images_for_each_label:
            min_count = min(min_count, len(images_for_each_label[label]))
        
        print(min_count)
        
        for label in images_for_each_label:
            print(label)
            for image_path, frame_num in images_for_each_label[label][:min_count]:
                self.image_path.append(image_path)
                self.labels.append(label)
                self.frame_num.append(frame_num)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))[16:16+224, 16:16+224]
        img = self.intensity_normalization(img)    
        
        img = torch.from_numpy(np.array([img] * 3))
#         img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)
    
class JpegT1T2ProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_file, mode):
        
        data_info = pd.read_csv(csv_file)
        data_info = data_info[data_info['mode'] == mode]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        model_mapping_dict = {'T1':0, 'T2':1}
        
        for idx, row in tqdm(data_info.iterrows()):
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(model_mapping_dict[row['Weighting']])
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}
    

class JpegT1_TETRTIProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        seriesIdentifier_to_TETRTI_label_dict = {}
        for idx, row in label_info.iterrows():
            seriesIdentifier_to_TETRTI_label_dict[row['seriesIdentifier']] = row['TETRTI_label']


        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_TETRTI_label_dict:
                continue
            label = seriesIdentifier_to_TETRTI_label_dict[row['seriesIdentifier']]
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}


class JpegT1_TETRTI_BalancedProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode, label_ratio_to_keep_dict):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        seriesIdentifier_to_TETRTI_label_dict = {}
        for idx, row in label_info.iterrows():
            seriesIdentifier_to_TETRTI_label_dict[row['seriesIdentifier']] = row['TETRTI_label']


        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_TETRTI_label_dict:
                continue
            label = seriesIdentifier_to_TETRTI_label_dict[row['seriesIdentifier']]
            if random.random() > label_ratio_to_keep_dict[label]:
                continue
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}


    
class JpegSeverityProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        seriesIdentifier_to_schwab_label_dict = {}
        model_mapping_dict = {50: 0, 60: 1, 65: 2, 70: 3, 75: 4, 80: 5, 85: 6, 90: 7, 95: 8, 100: 9}
        for idx, row in label_info.iterrows():
            seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']] = model_mapping_dict[row['schwab']]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_schwab_label_dict:
                continue
            label = seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']]
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}

    
class JpegSeveritySelectedLabelsProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        seriesIdentifier_to_schwab_label_dict = {}
        model_mapping_dict = {80: 0, 90: 1, 95: 2, 100: 3}
        for idx, row in label_info.iterrows():
            if row['schwab'] not in model_mapping_dict:
                continue
            seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']] = model_mapping_dict[row['schwab']]

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_schwab_label_dict:
                continue
            label = seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']]
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}

class ResnetJpegSeveritySelectedLabelsProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        seriesIdentifier_to_schwab_label_dict = {}
        model_mapping_dict = {80: 0, 90: 1, 95: 2, 100: 3}
        label_count = {80: 0, 90: 0, 95: 0, 100: 0}
        for idx, row in label_info.iterrows():
            if row['schwab'] not in model_mapping_dict:
                continue
            label_count[row['schwab']] += 1
            if label_count[row['schwab']] > 150:
                continue
            seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']] = model_mapping_dict[row['schwab']]
            

        self.image_path = []
        self.frame_num = []
        self.labels = []
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_schwab_label_dict:
                continue
            label = seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']]
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))[16:16+224, 16:16+224]
        
        img = torch.from_numpy(np.array([img] * 3))
#         img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx]}


class ResnetJpegSeveritySelectedLabelsWithNonImageFeaturesProcessedDataset(Dataset):
    """
    mode: either train, val, or test
    """
    def __init__(self, csv_info_file, csv_label_file, mode):
        
        data_info = pd.read_csv(csv_info_file)
        data_info = data_info[data_info['mode'] == mode]
        
        label_info = pd.read_csv(csv_label_file)
        
        selected_features = ['TE_x', 'TR_x','TI_x']
#         selected_features = ['Weighting_x', 'Pulse Sequence', 'TE_x', 'TR_x','TI_x', 'Manufacturer_x', 'Mfg Model']

        features_one_hot_dic = {}
        for feature in selected_features:
            features_one_hot_dic[feature] = self.create_one_hot_label(label_info[feature].unique())
            
        seriesIdentifier_to_features_dict = {}
        seriesIdentifier_to_schwab_label_dict = {}
        model_mapping_dict = {80: 0, 90: 1, 95: 2, 100: 3}
        for idx, row in label_info.iterrows():
            if row['schwab'] not in model_mapping_dict:
                continue
            seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']] = model_mapping_dict[row['schwab']]
            
            tmp_features = []
            for feature in selected_features:
                tmp_features += features_one_hot_dic[feature][row[feature]]
            seriesIdentifier_to_features_dict[row['seriesIdentifier']] = tmp_features
            
        
        self.image_path = []
        self.frame_num = []
        self.labels = []
        self.features = []
        
        for idx, row in tqdm(data_info.iterrows()):
            if row['seriesIdentifier'] not in seriesIdentifier_to_schwab_label_dict:
                continue
            label = seriesIdentifier_to_schwab_label_dict[row['seriesIdentifier']]
            self.image_path.append(row['preprocessed_2_path_jpeg'])
            self.labels.append(label)
            self.frame_num.append(row['frame_num'])
            self.features.append(seriesIdentifier_to_features_dict[row['seriesIdentifier']])  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))[16:16+224, 16:16+224]
        
        img = torch.from_numpy(np.array([img] * 3))
        features = torch.from_numpy(np.array(self.features[idx]))
#         img = torch.unsqueeze(img, 0)
        return {'image': img, 'label': self.labels[idx], 'image_path': self.image_path[idx], 'frame_num': self.frame_num[idx], 'features': features}

    def create_one_hot_label(self, data_list):
        mapping_dic = {}
        for i, item in enumerate(data_list):
            mapping_dic[item] = i
            
        mapping_oneHot_dic = {}
        k = len(mapping_dic)

        for item in mapping_dic:
            tmp = [0] * k 
            tmp[mapping_dic[item]] = 1
            mapping_oneHot_dic[item] = tmp

        return mapping_oneHot_dic
