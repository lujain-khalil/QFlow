import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

SAMPLING_ROOT = './sampling'
JSON_OUT = 'samples_evaluation_results.json'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

fid_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor()
])

inception_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
])

def get_real_loader(name):
    try:
        if name == 'mnist':
            ds = datasets.MNIST('./data', train=False, download=True, transform=fid_transform)
        elif name == 'fmnist':
            ds = datasets.FashionMNIST('./data', train=False, download=True, transform=fid_transform)
        elif name == 'cifar10':
            ds = datasets.CIFAR10('./data', train=False, download=True, transform=fid_transform)
        elif name == 'celebA':
            ds = datasets.CelebA('./data', split='test', download=False, transform=fid_transform)
        else:
            raise ValueError(f'Unknown dataset: {name}')
        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
    except (RuntimeError, FileNotFoundError, ConnectionError) as e:
        print(f"Error loading dataset {name}: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error loading dataset {name}: {str(e)}")
        raise


def compute_metrics(fake_paths, real_loader):
    """Compute FID and Inception Score on fake_paths vs real_loader."""

    fid = FrechetInceptionDistance(feature=2048).to(device)
    for real_imgs, _ in real_loader:
        real_uint8 = (real_imgs * 255).to(torch.uint8).to(device)
        fid.update(real_uint8, real=True)

    fake_fid_ds = [fid_transform(Image.open(p)) for p in fake_paths]
    fake_fid_loader = DataLoader(fake_fid_ds, batch_size=128, shuffle=False)
    for batch in fake_fid_loader:
        fake_uint8 = (batch * 255).to(torch.uint8).to(device)
        fid.update(fake_uint8, real=False)
    fid_score = fid.compute().item()

    # Inception Score (only needs fake)
    is_metric = InceptionScore().to(device)
    fake_is_ds = [inception_transform(Image.open(p)) for p in fake_paths]
    fake_is_loader = DataLoader(fake_is_ds, batch_size=128, shuffle=False)

    for batch in fake_is_loader:
        fake_uint8 = (batch * 255).to(torch.uint8).to(device)
        is_metric.update(fake_uint8)
    is_mean, is_std = is_metric.compute()

    return fid_score, is_mean.item(), is_std.item()

def find_sample_dirs(ds_root, ds):
    ds_dict = {}
    for method in os.listdir(ds_root):
        method_dir = os.path.join(ds_root, method)

        if method in ['ot', 'uniform']:
            method_dict = {}
            for bit_width in sorted(os.listdir(method_dir)):
                bit_width_dir = os.path.join(method_dir, bit_width)
                
                image_paths = []

                if ds != 'celebA':
                    for class_name in os.listdir(bit_width_dir):
                        class_dir = os.path.join(bit_width_dir, class_name)
                        image_paths.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')])
                else:
                    image_paths.extend([os.path.join(bit_width_dir, f) for f in os.listdir(bit_width_dir) if f.endswith('.png')])

                if len(image_paths) == 1000:
                    method_dict[bit_width] = image_paths
                else:
                    raise ValueError(f"Expected 1000 images for {bit_width} in {method}, but found {len(image_paths)}.")

            ds_dict[method] = method_dict    
        else:  # method == 'full'
            image_paths = []

            if ds != 'celebA':
                for class_name in os.listdir(method_dir):
                    class_dir = os.path.join(method_dir, class_name)
                    image_paths.extend([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.png')])
            else:
                    image_paths.extend([os.path.join(method_dir, f) for f in os.listdir(method_dir) if f.endswith('.png')])

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

def main():
    results = {}
    for ds in ('celebA', 'mnist','fmnist','cifar10'):
        ds_root = os.path.join(SAMPLING_ROOT, ds)
        if not os.path.isdir(ds_root):
            print(f"Warning: no samples for '{ds}' at {ds_root}, skipping")
            continue
        ds_dict = find_sample_dirs(ds_root, ds)
        real_loader = get_real_loader(ds)
        results[ds] = {}

        for method in ds_dict.keys():
            results[ds][method] = {}

            if method in ['ot', 'uniform']:
                for bitlabel, paths in ds_dict[method].items():
                    print(f"→ {ds} / {method} / {bitlabel}")
                    fps = sorted(f for f in paths if f.endswith('.png'))
                    fid, is_mean, is_std = compute_metrics(fps, real_loader)
                    print(f"  - FID: {fid:.5f}, IS mean: {is_mean:.5f}, IS std: {is_std:.5f}")
                    results[ds][method][bitlabel] = {
                        'fid': fid,
                        'is_mean': is_mean,
                        'is_std': is_std
                    }
            else:  # method == 'full'
                paths = ds_dict[method]
                print(f"→ {ds} / {method}")
                fps = sorted(f for f in paths if f.endswith('.png'))
                fid, is_mean, is_std = compute_metrics(fps, real_loader)
                print(f"  - FID: {fid:.5f}, IS mean: {is_mean:.5f}, IS std: {is_std:.5f}")
                results[ds][method] = {
                    'fid': fid,
                    'is_mean': is_mean,
                    'is_std': is_std
                }

    # save JSON
    with open(JSON_OUT, 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"\nDone! metrics written to {JSON_OUT}")

if __name__ == '__main__':
    main()