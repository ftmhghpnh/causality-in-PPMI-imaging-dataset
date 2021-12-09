import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import nibabel as nb
from torch.utils.data import Dataset
from PIL import Image


class JpegClusteringProcessedDataset(Dataset):
    
    def __init__(self, csv_file):
        
        data_info = pd.read_csv(csv_file)
        self.image_path = []
        self.seriesIdentifier = []
        
        for index, row in data_info.iterrows():
            if row['frame_num'] == 128: ### based on the distribution of the images, decision has been made
                self.image_path.append(row['preprocessed_2_path_jpeg'])
                self.seriesIdentifier.append(row['seriesIdentifier'])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        
        img = np.asarray(Image.open(self.image_path[idx]))
        img = self.padding_to_ideal_shape(img)
        img = self.intensity_normalization(img)  
        
        
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        return {'image': img, 'image_path': self.image_path[idx], 'seriesIdentifier': self.seriesIdentifier[idx]}
    
    def intensity_normalization(self, raw):
        return (raw - np.mean(raw)) / np.std(raw)
    
    def padding_to_ideal_shape(self, img, ideal_shape=(256, 256)):
        img_shape = img.shape
        i_shape = int((256 - img_shape[0])/2.0)
        j_shape = int((256 - img_shape[1])/2.0)
        marginal_zero = np.zeros(ideal_shape)
        marginal_zero[i_shape:i_shape+img_shape[0], j_shape:j_shape+img_shape[1]] = img
        return marginal_zero[3:-3, 3:-3]

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        #in_c, out_c, kernel_size
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(29 * 29 * 128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 29 * 29 * 128),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(128, 29, 29))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 5, stride=2, 
            padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    
batch_size = 32
jpeg_csv_path = '/w/246/gzk/PPMI/codes/T1_PD_clustring.csv'

dataset = JpegClusteringProcessedDataset(jpeg_csv_path)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Initialize the two networks
d = 256

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d)
decoder = Decoder(encoded_space_dim=d)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


iteration_loss_csv_path = 'auto_encoder_loss_per_iteration.csv'
with open(iteration_loss_csv_path, 'w') as f:
    f.write('epoch,train_loss\n')
    
### Training function
def train_epoch(epoch, encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for data in dataloader:
        image_batch = data['image'].to(device, dtype=torch.float)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        with open(iteration_loss_csv_path, 'w') as f:
            f.write('{},{}\n'.format(epoch, loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for data in dataloader:
            image_batch = data['image'].to(device, dtype=torch.float)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = dataset[i]['image'].unsqueeze(0).to(device, dtype=torch.float)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img  = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.show()
    
loss_csv_path = 'auto_encoder_loss_per_epoch.csv'
with open(loss_csv_path, 'w') as f:
    f.write('epoch,train_loss,val_loss\n')

num_epochs = 30
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
    train_loss =train_epoch(epoch, encoder,decoder,device,data_loader,loss_fn,optim)
    val_loss = test_epoch(encoder,decoder,device,data_loader,loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)
    with open(loss_csv_path, 'a') as f:
        f.write('{},{},{}\n'.format(epoch, train_loss, val_loss))
#     plot_ae_outputs(encoder,decoder,n=5)