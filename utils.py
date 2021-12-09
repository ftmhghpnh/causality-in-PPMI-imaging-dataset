import os
import torch
import progressbar
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def validation(model, val_loader, criterion, device, predictions_csv_path=None):
    '''
    Helper validation loss and accuracy function for use during model training
    Args:
        model: model object (ex: vgg16, resnet or custom model using nn.Module)
        val_loader: data loader for validation set
        criterion: loss function used for training
        device: cuda or cpu device
    Returns:
        test_loss: the loss during validation testing
        accuracy: the accuracy during validation testing
    '''
    test_loss = 0
    total = 0
    correct = 0
    true_labels = []
    preds = []
    image_paths = []
    frame_nums = []
    with progressbar.ProgressBar(max_value=len(val_loader)) as bar:
        cnt = 0
        for data in val_loader:
            X, y = data['image'].to(device, dtype=torch.float), data['label'].to(device, dtype=torch.long)
            output = model(X)
            test_loss += criterion(output, y).item()

            correct += torch.sum(output.argmax(axis=1) == y).item()
            total += output.shape[0]
            
            true_labels += y.tolist()
            preds += output.tolist()
            image_paths += data['image_path']
            frame_nums += data['frame_num']
            
            cnt += 1
            bar.update(cnt)

    accuracy = correct/total

    if predictions_csv_path:
        with open(predictions_csv_path, 'w') as f:
            f.write('image_path,frame_num,true_label,pred\n')
            for i in range(len(image_paths)):
                f.write('{},{},{},{}\n'.format(image_paths[i], frame_nums[i], true_labels[i], preds[i]))

    return test_loss, accuracy



def train(model, train_loader, val_loader, optimizer, criterion, epochs, csv_result_path, save_path, predictions_path, device, scheduler = None):

    with open(csv_result_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
    for epoch in range(epochs):
        model.train()
        with progressbar.ProgressBar(max_value=len(train_loader)) as bar:
            cnt = 0
            for data in train_loader:
                X, y = data['image'].to(device, dtype=torch.float), data['label'].to(device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                cnt += 1
                bar.update(cnt)
        if scheduler is not None:   
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            train_predictions_csv_path = os.path.join(predictions_path, 'train_predictions_{}.csv'.format(epoch))
            train_loss, train_acc = validation(model, train_loader, criterion, device, train_predictions_csv_path)
            val_predictions_csv_path = os.path.join(predictions_path, 'val_predictions_{}.csv'.format(epoch))
            val_loss, val_acc = validation(model, val_loader, criterion, device, val_predictions_csv_path)
            
        print("Training Loss:", train_loss)
        print("Training Accuracy:", train_acc)
        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_acc)
        
        torch.save(model.state_dict(), save_path+str(epoch)+'.pth')
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        }, save_path+'_ckpt_'+str(epoch)+'.pth')
        
        with open(csv_result_path, 'a') as f:
            f.write('{},{},{},{},{}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc))
            
            
def validation_with_features(model, val_loader, criterion, device, predictions_csv_path=None):
    '''
    Helper validation loss and accuracy function for use during model training
    Args:
        model: model object (ex: vgg16, resnet or custom model using nn.Module)
        val_loader: data loader for validation set
        criterion: loss function used for training
        device: cuda or cpu device
    Returns:
        test_loss: the loss during validation testing
        accuracy: the accuracy during validation testing
    '''
    test_loss = 0
    total = 0
    correct = 0
    true_labels = []
    preds = []
    image_paths = []
    frame_nums = []
    with progressbar.ProgressBar(max_value=len(val_loader)) as bar:
        cnt = 0
        for data in val_loader:
            X, features, y = data['image'].to(device, dtype=torch.float), data['features'].to(device, dtype=torch.float), data['label'].to(device, dtype=torch.long)
            output = model(X, features)
            test_loss += criterion(output, y).item()

            correct += torch.sum(output.argmax(axis=1) == y).item()
            total += output.shape[0]
            
            true_labels += y.tolist()
            preds += output.tolist()
            image_paths += data['image_path']
            frame_nums += data['frame_num']
            
            cnt += 1
            bar.update(cnt)

    accuracy = correct/total

    if predictions_csv_path:
        with open(predictions_csv_path, 'w') as f:
            f.write('image_path,frame_num,true_label,pred\n')
            for i in range(len(image_paths)):
                f.write('{},{},{},{}\n'.format(image_paths[i], frame_nums[i], true_labels[i], preds[i]))

    return test_loss, accuracy


def train_with_features(model, train_loader, val_loader, optimizer, criterion, epochs, csv_result_path, save_path, predictions_path, device, scheduler = None):

    with open(csv_result_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc\n')
    for epoch in range(epochs):
        model.train()
        with progressbar.ProgressBar(max_value=len(train_loader)) as bar:
            cnt = 0
            for data in train_loader:
                X, features, y = data['image'].to(device, dtype=torch.float), data['features'].to(device, dtype=torch.float), data['label'].to(device, dtype=torch.long)
                optimizer.zero_grad()
                output = model(X, features)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                cnt += 1
                bar.update(cnt)
        if scheduler is not None:   
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            train_predictions_csv_path = os.path.join(predictions_path, 'train_predictions_{}.csv'.format(epoch))
            train_loss, train_acc = validation_with_features(model, train_loader, criterion, device, train_predictions_csv_path)
            val_predictions_csv_path = os.path.join(predictions_path, 'val_predictions_{}.csv'.format(epoch))
            val_loss, val_acc = validation_with_features(model, val_loader, criterion, device, val_predictions_csv_path)
            
        print("Training Loss:", train_loss)
        print("Training Accuracy:", train_acc)
        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_acc)
        
        torch.save(model.state_dict(), save_path+str(epoch)+'.pth')
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        }, save_path+'_ckpt_'+str(epoch)+'.pth')
        
        with open(csv_result_path, 'a') as f:
            f.write('{},{},{},{},{}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc))
