import os 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import timm
from PIL import Image
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DATA_MODELS = {
    'fmnist': 'inceptionv3',
    'mnist': 'resnet18',
    'cifar10': 'bit'
}

INPUT_SIZES = {
    'fmnist': 28,
    'mnist': 28,
    'cifar10': 32
}

def load_model(dataset):
    model_name = DATA_MODELS[dataset]
    
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
        
    checkpoint = torch.load(f'./classifiers/{dataset}_{model_name}.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model

def get_transform(dataset, input_size):
    if dataset in ['mnist', 'fmnist']:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    return transform

def find_sample_dirs(ds_root, ds):
    if ds == 'celebA':
        raise ValueError("CelebA dataset is not supported in this script.")
    
    ds_dict = {}
    for method in os.listdir(ds_root):
        method_dir = os.path.join(ds_root, method)

        if method in ['ot', 'uniform']:
            method_dict = {}
            for bit_width in sorted(os.listdir(method_dir)):
                bit_width_dir = os.path.join(method_dir, bit_width)
                
                image_paths = []

                for class_name in os.listdir(bit_width_dir):
                    class_dir = os.path.join(bit_width_dir, class_name)
                    image_paths.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')])
                
                if len(image_paths) == 1000:
                    method_dict[bit_width] = image_paths
                else:
                    raise ValueError(f"Expected 1000 images for {bit_width} in {method}, but found {len(image_paths)}.")

            ds_dict[method] = method_dict       
        else: # method == 'full'
            image_paths = []

            for class_name in os.listdir(method_dir):
                class_dir = os.path.join(method_dir, class_name)
                image_paths.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')])

            if len(image_paths) == 1000:
                ds_dict[method] = image_paths
            else:
                raise ValueError(f"Expected 1000 images for {method}, but found {len(image_paths)}.")
    
    print(f"Found {len(ds_dict)} sampling methods for {ds}:")
    for method in ds_dict.keys():
        if method in ['ot', 'uniform']:
            bit_widths = ds_dict[method]
            print(f"  - {method}: {len(bit_widths)} bit widths")
            for bit_width, paths in bit_widths.items():
                print(f"    - {bit_width}: {len(paths)} images")
        else:
            paths = ds_dict[method]
            print(f"  - {method}: {len(paths)} images")
            
    return ds_dict

def evaluate_images(model, image_paths, transform, dataset):
    predictions = []
    true_labels = []
    
    for path in image_paths:
        # Extract true label from directory name
        true_label = int(os.path.basename(os.path.dirname(path)))
        true_labels.append(true_label)
        
        # Load and preprocess image
        img = Image.open(path)
        img = transform(img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
    
    return predictions, true_labels

def calculate_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)

    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

if __name__ == "__main__":
    samples_root = "./sampling"
    datasets = ['mnist', 'fmnist', 'cifar10']

    results = {}
    
    for ds in datasets:
        print(f"Evaluating {ds} dataset...")
        ds_dict = find_sample_dirs(os.path.join(samples_root, ds), ds)
        
        model = load_model(ds)
        transform = get_transform(ds, INPUT_SIZES[ds])
        
        ds_results = {}
        
        for method in ds_dict.keys():
            if method in ['ot', 'uniform']:
                bit_widths = ds_dict[method]
                method_results = {}
                
                for bit_width, image_paths in bit_widths.items():
                    print(f"Processing {method} - {bit_width}...")
                    
                    predictions, true_labels = evaluate_images(model, image_paths, transform, ds)
                    metrics = calculate_metrics(predictions, true_labels)
                    
                    method_results[bit_width] = metrics
                
                ds_results[method] = method_results
            else: # method == 'full'
                image_paths = ds_dict[method]
                print(f"Processing {method}...")
                
                predictions, true_labels = evaluate_images(model, image_paths, transform, ds)
                metrics = calculate_metrics(predictions, true_labels)
                
                ds_results[method] = metrics
            
        results[ds] = ds_results

    with open('downstream_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\nEvaluation complete! Results saved to downstream_results.json")

