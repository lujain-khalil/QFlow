import json 
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Read the JSON files
with open('downstream_results.json', 'r') as f:
    downstream_results = json.load(f)

with open('samples_evaluation_results.json', 'r') as f:
    samples_results = json.load(f)

# Bit width mapping and order
bit_mapping = {
    'full': 'FP32',
    'uniform_16bit': 'INT16',
    'uniform_8bit': 'INT8',
    'uniform_4bit': 'INT4',
    'uniform_2bit': 'INT2',
    'ot_16bit': 'INT16',
    'ot_8bit': 'INT8',
    'ot_4bit': 'INT4',
    'ot_2bit': 'INT2'
}

x_order = ['FP32', 'INT16', 'INT8', 'INT4', 'INT2']

def plot_metric(dataset, metric_name, metric_data_uniform, metric_data_ot, ylabel):
    plt.figure(figsize=(6, 4))
    
    # Extract x and y values for uniform quantization
    x_uniform = []
    y_uniform = []
    # Add FP32 (full precision) point
    if 'full' in metric_data_uniform:
        x_uniform.append('FP32')
        y_uniform.append(metric_data_uniform['full'])
    # Add quantized points
    for width in ['16', '8', '4', '2']:
        key = f'uniform_{width}bit'
        if key in metric_data_uniform:
            x_uniform.append(f'INT{width}')
            y_uniform.append(metric_data_uniform[key])
    
    # Extract x and y values for OT quantization
    x_ot = []
    y_ot = []
    # Add FP32 (full precision) point
    if 'full' in metric_data_ot:
        x_ot.append('FP32')
        y_ot.append(metric_data_ot['full'])
    # Add quantized points
    for width in ['16', '8', '4', '2']:
        key = f'ot_{width}bit'
        if key in metric_data_ot:
            x_ot.append(f'INT{width}')
            y_ot.append(metric_data_ot[key])
    
    plt.plot(x_uniform, y_uniform, label='Uniform',                     
                    color='cyan',
                    linestyle='--',
                    marker='o',
                    markersize = 5,
)
    plt.plot(x_ot, y_ot, label='OT-based (Proposed)',                     
                    color='purple',
                    linestyle='-',
                    marker='*',
                    markersize = 10,
)
    
    plt.xlabel('Quantization')
    plt.ylabel(ylabel)
    plt.title(f'{dataset} - {metric_name}')
    plt.grid(True)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{dataset.lower()}_{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

# Plot classifier accuracy for MNIST, FMNIST, CIFAR10
datasets = ['mnist', 'fmnist', 'cifar10']
for dataset in datasets:
    # Accuracy plots
    uniform_acc = {}
    uniform_acc['full'] = downstream_results[dataset]['full']['accuracy']
    uniform_acc.update({k: v['accuracy'] for k, v in downstream_results[dataset]['uniform'].items()})
    
    ot_acc = {}
    ot_acc['full'] = downstream_results[dataset]['full']['accuracy']
    ot_acc.update({k: v['accuracy'] for k, v in downstream_results[dataset]['ot'].items()})
    plot_metric(dataset.upper(), 'Classifier Accuracy', uniform_acc, ot_acc, 'Accuracy')
    
    # Inception Score plots
    uniform_is = {}
    uniform_is['full'] = samples_results[dataset]['full']['is_mean']
    uniform_is.update({k: v['is_mean'] for k, v in samples_results[dataset]['uniform'].items()})
    
    ot_is = {}
    ot_is['full'] = samples_results[dataset]['full']['is_mean']
    ot_is.update({k: v['is_mean'] for k, v in samples_results[dataset]['ot'].items()})
    plot_metric(dataset.upper(), 'Inception Score', uniform_is, ot_is, 'IS Mean')
    
    # FID plots
    uniform_fid = {}
    uniform_fid['full'] = samples_results[dataset]['full']['fid']
    uniform_fid.update({k: v['fid'] for k, v in samples_results[dataset]['uniform'].items()})
    
    ot_fid = {}
    ot_fid['full'] = samples_results[dataset]['full']['fid']
    ot_fid.update({k: v['fid'] for k, v in samples_results[dataset]['ot'].items()})
    plot_metric(dataset.upper(), 'FID Score', uniform_fid, ot_fid, 'FID')

# Plot CelebA FID and Inception Score
# FID plot
uniform_fid = {}
uniform_fid['full'] = samples_results['celebA']['full']['fid']
uniform_fid.update({k: v['fid'] for k, v in samples_results['celebA']['uniform'].items()})

ot_fid = {}
ot_fid['full'] = samples_results['celebA']['full']['fid']
ot_fid.update({k: v['fid'] for k, v in samples_results['celebA']['ot'].items()})
plot_metric('CelebA', 'FID Score', uniform_fid, ot_fid, 'FID')

# Inception Score plot
uniform_is = {}
uniform_is['full'] = samples_results['celebA']['full']['is_mean']
uniform_is.update({k: v['is_mean'] for k, v in samples_results['celebA']['uniform'].items()})

ot_is = {}
ot_is['full'] = samples_results['celebA']['full']['is_mean']
ot_is.update({k: v['is_mean'] for k, v in samples_results['celebA']['ot'].items()})
plot_metric('CelebA', 'Inception Score', uniform_is, ot_is, 'IS Mean')