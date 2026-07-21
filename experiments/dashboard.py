import os
import json
import glob
import torch
import matplotlib.pyplot as plt
from typing import List

from dotenv import load_dotenv
load_dotenv()

def load_experiment_results(base_dir: str) -> List[dict]:
    results = []
    pattern = os.path.join(base_dir, 'experiments', 'results', '*', 'summary.json')
    for summary_path in glob.glob(pattern):
        with open(summary_path, 'r') as f:
            results.append(json.load(f))
    return results

def plot_comparison_loss_curves(results_dir: str, base_dir: str):
    plt.figure(figsize=(10, 6))
    
    pattern = os.path.join(base_dir, 'experiments', 'results', '*', 'summary.json')
    for summary_path in glob.glob(pattern):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            
        checkpoint_name = summary['checkpoint_name']
        checkpoint_path = os.path.join(base_dir, 'bin', checkpoint_name)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                train_loss = history.train_loss
                epochs = list(range(1, len(train_loss) + 1))
                plt.plot(epochs, train_loss, label=summary['name'])
                
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves Comparison')
    plt.legend()
    plt.grid(True)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comparison_loss_curves.png'))
    plt.close()

def plot_comparison_bar_chart(results: List[dict], results_dir: str):
    if not results:
        return
        
    names = [r['name'] for r in results]
    test_losses = [r['final_test_loss'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, test_losses, color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Final Test Loss')
    plt.title('Final Test Loss by Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comparison_test_loss.png'))
    plt.close()

def plot_parameter_comparison(results: List[dict], results_dir: str):
    if not results:
        return
        
    names = [r['name'] for r in results]
    params = [r['param_count'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, params, color='lightgreen')
    plt.xlabel('Experiment')
    plt.ylabel('Parameter Count')
    plt.title('Parameter Count by Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comparison_params.png'))
    plt.close()

def generate_dashboard(base_dir: str):
    results = load_experiment_results(base_dir)
    results_dir = os.path.join(base_dir, 'experiments', 'results')
    
    plot_comparison_loss_curves(results_dir, base_dir)
    plot_comparison_bar_chart(results, results_dir)
    plot_parameter_comparison(results, results_dir)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_dashboard(base_dir)
