import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import wandb
import random
import numpy as np

def get_data_transforms(args):

    return {
        'train': transforms.Compose([
            transforms.Resize(args.resize_size),  
            transforms.RandomResizedCrop(args.crop_size),  
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ]),
        'valid': transforms.Compose([
            transforms.Resize(args.resize_size),  
            transforms.CenterCrop(args.crop_size),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ]),
        'test': transforms.Compose([
            transforms.Resize(args.resize_size),  
            transforms.CenterCrop(args.crop_size),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ]),
    }

def get_dataloaders(args):
    data_transforms = get_data_transforms(args)
    
    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms['train'])
    valid_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'valid'), data_transforms['valid'])
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'test'), data_transforms['test'])
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    dataset_sizes = {
        'train': len(train_dataset),
        'valid': len(valid_dataset),
        'test': len(test_dataset)
    }
    class_names = train_dataset.classes
    
    return train_loader, valid_loader, test_loader, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, train_loader, valid_loader, dataset_sizes, device, num_epochs=25, patience=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc, "epoch": epoch})

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Classification")
    parser.add_argument('--data_dir', default='/home/cal-05/hj/ND_CL/Classifier/dataset/v1', type=str, help='Dataset directory')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--resize_size', default=256, type=int, help='Batch size')
    parser.add_argument('--crop_size', default=224, type=int, help='Batch size')
    parser.add_argument('--patience', default=10, type=int, help='Batch size')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    # wandb 초기화
    wandb.login()
    wandb.init(
    project='car-classification',
    name=f'epochs_{args.num_epochs}_batch_{args.batch_size}_lr_{args.learning_rate}_resize_{args.resize_size}_crop_{args.crop_size}',
    config={
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'resize_size': args.resize_size,
        'crop_size': args.crop_size,
        'patience': args.patience
    }
)
    wandb.init(project='car-classification', save_code=True)

    train_loader, valid_loader, test_loader, dataset_sizes, class_names = get_dataloaders(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader, valid_loader, dataset_sizes, device, args.num_epochs, args.patience)

    # 모델 가중치 저장
    torch.save(model_ft.state_dict(), 'best_model.pth')

    print(f"Model saved to 'best_model.pth'")

    # test
    model_ft.eval()
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Acc: {test_acc:.4f}')

    wandb.log({"test_acc": test_acc})


if __name__ == "__main__":
    main()
