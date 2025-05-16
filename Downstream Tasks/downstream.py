import argparse
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import timm
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_MODELS = {
    'fmnist': 'inceptionv3',
    'mnist': 'resnet18',
    'cifar10': 'bit'
}

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 100
WD = 5e-4
MOMENTUM = 0.9

def get_dataset_and_loader(dataset_name, data_dir, batch_size, train = True):

    if dataset_name == 'fmnist':
        input_size = 299
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(15, translate=(0.1,0.1)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02,0.15)),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        )
    elif dataset_name == 'mnist':
        input_size = 224
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.02,0.02)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        )
    elif dataset_name == 'cifar10':
        input_size = 224
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        )
    else:
        raise ValueError(f"Unknown model: {DATA_MODELS[dataset_name]}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)


def load_model(model_name, device):
    if model_name == 'inceptionv3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.aux_logits = False
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
    elif model_name == 'bit':
        model = timm.create_model('resnetv2_101x1_bitm', pretrained=True, num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct, total, running_loss = 0.0, 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if criterion:
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

    accuracy = correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss

def fine_tune(model, train_loader, device, model_name):
    for p in model.parameters():
        p.requires_grad = False
    
    head = model.fc if hasattr(model, 'fc') else model.head
    for p in head.parameters():
        p.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            for p in m.parameters():
                p.requires_grad = True

    if model_name == 'bit':
        for p in model.stages[-1].parameters():
            p.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Training {num_trainable_params} parameters for {EPOCHS} epochs")

    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=LR, 
                                  weight_decay=WD)
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    final_loss = None
    final_acc = None

    for epoch in range(EPOCHS):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        i = 0
        for imgs, labels in train_loader:
            if (i+1) % 10 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{EPOCHS} - Image {i+1}/{len(train_loader)}")

            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / len(train_loader))
            i += 1

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item() * imgs.size(0)
            
        final_loss = running_loss / len(train_loader.dataset)
        final_acc = correct / total
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1}/{EPOCHS} - Loss: {final_loss:.4f}, Training Accuracy: {final_acc:.2f}")

    return num_trainable_params, final_loss, final_acc

def main():

    parser = argparse.ArgumentParser(description='Load, save, and evaluate pretrained classifiers.')
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'cifar10'], required=True,
                        help='Name of the model to load and evaluate')
    args = parser.parse_args()

    DATA_DIR = './data'
    SAVE_DIR = './models'
    RESULTS_DIR = './results'
    BATCH_SIZE = 128

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading dataset: {args.dataset}")
    train_loader = get_dataset_and_loader(args.dataset, DATA_DIR, BATCH_SIZE, train=True)
    test_loader  = get_dataset_and_loader(args.dataset, DATA_DIR, BATCH_SIZE, train=False)

    model_name = DATA_MODELS[args.dataset]
    model = load_model(model_name, device)

    trainable_params, train_loss, train_acc = fine_tune(model, train_loader, device, model_name)

    save_path = os.path.join(SAVE_DIR, f"{args.dataset}_{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved fine-tuned weights to {save_path}")

    test_acc, test_loss = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}, Test Loss: {test_loss:.4f}")

    metrics = {
        'model': model_name,
        'dataset': args.dataset,
        'trainable_params': trainable_params,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
    }
    
    metrics_file = os.path.join(RESULTS_DIR, f"{args.dataset}_{model_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

if __name__ == '__main__':
    main()