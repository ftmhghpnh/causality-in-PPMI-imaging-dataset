import os
import torch
import torch.nn as nn
import torch.optim as optim

from processed_dataset import *
from torch.utils.data import DataLoader
from models import *
from utils import *

batch_size = 32
num_workers = 4
epochs = 10

# csv_path = '/w/246/gzk/PPMI/codes/first_model_jpeg_csv_info.csv'
csv_path = '/w/246/gzk/PPMI/codes/T1_T2_pdsubject_jpeg_info.csv'
csv_label_path = '/w/246/gzk/PPMI/codes/T1_severity_score_info_csv.csv'
# results_base_path = '/w/246/gzk/PPMI/results/v4/'
results_base_path = '/w/284/gzk/result/T1_severity_with_features/v13'

if not os.path.exists(results_base_path):
    os.makedirs(results_base_path)

csv_result_path = os.path.join(results_base_path, 'model_v13.csv')
save_path = os.path.join(results_base_path, 'model_v13_')
predictions_csv_path = os.path.join(results_base_path, 'preds')
if not os.path.exists(predictions_csv_path):
    os.mkdir(predictions_csv_path)

# label_ratio_to_keep_dict = {
#     0: 1,
#     1: 1,
#     2: 0.5,
#     3: 0.5,
#     4: 0.1,
#     5: 1,
#     6: 1,
#     7: 1,
#     8: 1,
#     9: 1,
#     10: 1
# }
train_ds = ResnetJpegSeveritySelectedLabelsWithNonImageFeaturesProcessedDataset(csv_path, csv_label_path, 'train')
val_ds = ResnetJpegSeveritySelectedLabelsWithNonImageFeaturesProcessedDataset(csv_path, csv_label_path, 'val')

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet18WithNonImageFeaturesClassifier(num_classes=4, freeze_weights=False, features_count=212)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

train_with_features(model, train_loader, val_loader, optimizer, criterion, epochs, csv_result_path, save_path, predictions_csv_path, device)