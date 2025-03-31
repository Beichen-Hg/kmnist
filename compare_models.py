#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tool for comparing CNN and ResNet model performance

This script compares the performance of CNN and ResNet models trained on the Kuzushiji-MNIST dataset.
It loads training history data for both models and generates comprehensive comparison charts including:
1. Single model hyperparameter experiment analysis (loss functions, batch sizes, learning rates)
2. Overall performance comparison between models
3. Training efficiency and convergence speed comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

def load_json_file(filepath):
    """Load JSON file and return data"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load file {filepath}: {e}")
        return None

def compare_models(cnn_history_file, resnet_history_file, output_filename, output_dir=None):
    """
    Compare training history of two models and generate comparison charts
    
    Parameters:
        cnn_history_file: Path to CNN model history file
        resnet_history_file: Path to ResNet model history file
        output_filename: Output chart filename
        output_dir: Output directory, defaults to current directory
    """
    # If no output directory is specified, use current directory
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create presentation subdirectory
    presentation_dir = os.path.join(output_dir, 'presentation')
    if not os.path.exists(presentation_dir):
        os.makedirs(presentation_dir)
    
    # Load history data
    with open(cnn_history_file, 'r') as f:
        cnn_history = json.load(f)
    
    with open(resnet_history_file, 'r') as f:
        resnet_history = json.load(f)
    
    # Get common training epochs
    epochs_range = range(1, min(len(cnn_history['loss']), len(resnet_history['loss'])) + 1)
    
    # 1. Generate overall performance comparison chart
    plt.figure(figsize=(16, 12))
    
    # Compare training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, cnn_history['loss'][:len(epochs_range)], label='CNN Training Loss', linewidth=2)
    plt.plot(epochs_range, resnet_history['loss'][:len(epochs_range)], label='ResNet Training Loss', linewidth=2)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs_range)
    
    # Compare validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, cnn_history['val_loss'][:len(epochs_range)], label='CNN Validation Loss', linewidth=2)
    plt.plot(epochs_range, resnet_history['val_loss'][:len(epochs_range)], label='ResNet Validation Loss', linewidth=2)
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs_range)
    
    # Compare training accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, cnn_history['accuracy'][:len(epochs_range)], label='CNN Training Accuracy', linewidth=2)
    plt.plot(epochs_range, resnet_history['accuracy'][:len(epochs_range)], label='ResNet Training Accuracy', linewidth=2)
    plt.title('Training Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs_range)
    
    # Compare validation accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, cnn_history['val_accuracy'][:len(epochs_range)], label='CNN Validation Accuracy', linewidth=2)
    plt.plot(epochs_range, resnet_history['val_accuracy'][:len(epochs_range)], label='ResNet Validation Accuracy', linewidth=2)
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs_range)
    
    plt.suptitle('CNN vs ResNet Model Performance Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
    
    # Save chart
    output_path = os.path.join(presentation_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Model comparison chart saved to: {output_path}")
    
    # Calculate and print final performance metrics
    final_cnn_acc = cnn_history['val_accuracy'][-1]
    final_resnet_acc = resnet_history['val_accuracy'][-1]
    final_cnn_loss = cnn_history['val_loss'][-1]
    final_resnet_loss = resnet_history['val_loss'][-1]
    
    print("\n===== Final Model Performance Comparison =====")
    print(f"CNN Final Validation Accuracy: {final_cnn_acc:.4f}")
    print(f"ResNet Final Validation Accuracy: {final_resnet_acc:.4f}")
    print(f"CNN Final Validation Loss: {final_cnn_loss:.4f}")
    print(f"ResNet Final Validation Loss: {final_resnet_loss:.4f}")
    
    # Load hardware info files to compare training time
    try:
        cnn_hw_info_path = os.path.join(os.path.dirname(cnn_history_file), 'hardware_info.json')
        resnet_hw_info_path = os.path.join(os.path.dirname(resnet_history_file), 'resnet_hardware_info.json')
        
        with open(cnn_hw_info_path, 'r') as f:
            cnn_hw_info = json.load(f)
        
        with open(resnet_hw_info_path, 'r') as f:
            resnet_hw_info = json.load(f)
        
        print(f"\nCNN Average Time per Epoch: {cnn_hw_info['final_model_avg_epoch_time']:.2f}s ({cnn_hw_info['device']})")
        print(f"ResNet Average Time per Epoch: {resnet_hw_info['final_model_avg_epoch_time']:.2f}s ({resnet_hw_info['device']})")
        
        # Compare hyperparameters
        print("\n===== Hyperparameter Comparison =====")
        print(f"CNN: Loss={cnn_hw_info['selected_loss']}, Batch Size={cnn_hw_info['selected_batch_size']}, Learning Rate={cnn_hw_info['selected_learning_rate']}")
        print(f"ResNet: Loss={resnet_hw_info['selected_loss']}, Batch Size={resnet_hw_info['selected_batch_size']}, Learning Rate={resnet_hw_info['selected_learning_rate']}")
        
        # 2. Generate training time comparison chart
        plt.figure(figsize=(10, 6))
        models = ['CNN', 'ResNet']
        times = [cnn_hw_info['final_model_avg_epoch_time'], resnet_hw_info['final_model_avg_epoch_time']]
        devices = [cnn_hw_info['device'], resnet_hw_info['device']]
        
        bars = plt.bar(models, times, color=['#3498db', '#e74c3c'])
        plt.title('Model Training Time Comparison', fontsize=16)
        plt.ylabel('Average Time per Epoch (seconds)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels and device info to bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s\n({devices[i]})',
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(presentation_dir, 'training_time_comparison.png'), dpi=300)
        plt.close()
        
        # 3. Generate final performance comparison chart
        plt.figure(figsize=(12, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        acc_data = [final_cnn_acc, final_resnet_acc]
        acc_bars = plt.bar(models, acc_data, color=['#3498db', '#e74c3c'])
        plt.title('Final Validation Accuracy Comparison', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0.9, 1.0)  # Adjust Y-axis range to highlight differences
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, bar in enumerate(acc_bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12)
        
        # Loss comparison
        plt.subplot(1, 2, 2)
        loss_data = [final_cnn_loss, final_resnet_loss]
        loss_bars = plt.bar(models, loss_data, color=['#3498db', '#e74c3c'])
        plt.title('Final Validation Loss Comparison', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, bar in enumerate(loss_bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.suptitle('CNN vs ResNet Final Performance Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(presentation_dir, 'final_performance_comparison.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Could not load hardware info files or generate comparison charts: {e}")
    
    # 4. Analyze hyperparameter effects
    analyze_hyperparameters(os.path.dirname(cnn_history_file), os.path.dirname(resnet_history_file), 
                           presentation_dir)

def analyze_hyperparameters(cnn_dir, resnet_dir, output_dir):
    """Analyze hyperparameter effects on model performance"""
    print("\n===== Analyzing Hyperparameter Effects on Model Performance =====")
    
    # 1. Analyze loss function effects
    analyze_loss_functions(cnn_dir, resnet_dir, output_dir)
    
    # 2. Analyze batch size effects
    analyze_batch_sizes(cnn_dir, resnet_dir, output_dir)
    
    # 3. Analyze learning rate effects
    analyze_learning_rates(cnn_dir, resnet_dir, output_dir)

def analyze_loss_functions(cnn_dir, resnet_dir, output_dir):
    """Analyze loss function effects on model performance"""
    loss_functions = ['categorical_crossentropy', 'mean_squared_error']
    
    # Load CNN loss function experiment data
    cnn_loss_results = {}
    for loss_fn in loss_functions:
        history_file = os.path.join(cnn_dir, f'{loss_fn}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                cnn_loss_results[loss_fn] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1]
                }
    
    # Load ResNet loss function experiment data
    resnet_loss_results = {}
    for loss_fn in loss_functions:
        history_file = os.path.join(resnet_dir, f'resnet_{loss_fn}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                resnet_loss_results[loss_fn] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1]
                }
    
    # If enough data is available, generate loss function comparison chart
    if cnn_loss_results and resnet_loss_results:
        plt.figure(figsize=(14, 6))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(loss_functions))
        width = 0.35
        
        cnn_acc = [cnn_loss_results.get(loss_fn, {}).get('val_accuracy', 0) for loss_fn in loss_functions]
        resnet_acc = [resnet_loss_results.get(loss_fn, {}).get('val_accuracy', 0) for loss_fn in loss_functions]
        
        plt.bar(x - width/2, cnn_acc, width, label='CNN', color='#3498db')
        plt.bar(x + width/2, resnet_acc, width, label='ResNet', color='#e74c3c')
        
        plt.xlabel('Loss Function', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy with Different Loss Functions', fontsize=14)
        plt.xticks(x, [loss_fn.replace('_', '\n') for loss_fn in loss_functions])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, v in enumerate(cnn_acc):
            plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        for i, v in enumerate(resnet_acc):
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        # Loss comparison
        plt.subplot(1, 2, 2)
        cnn_loss = [cnn_loss_results.get(loss_fn, {}).get('val_loss', 0) for loss_fn in loss_functions]
        resnet_loss = [resnet_loss_results.get(loss_fn, {}).get('val_loss', 0) for loss_fn in loss_functions]
        
        plt.bar(x - width/2, cnn_loss, width, label='CNN', color='#3498db')
        plt.bar(x + width/2, resnet_loss, width, label='ResNet', color='#e74c3c')
        
        plt.xlabel('Loss Function', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss with Different Loss Functions', fontsize=14)
        plt.xticks(x, [loss_fn.replace('_', '\n') for loss_fn in loss_functions])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, v in enumerate(cnn_loss):
            plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        for i, v in enumerate(resnet_loss):
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        plt.suptitle('Effect of Loss Functions on Model Performance', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'loss_function_comparison.png'), dpi=300)
        plt.close()
        
        print("Loss function comparison chart saved")

def analyze_batch_sizes(cnn_dir, resnet_dir, output_dir):
    """Analyze batch size effects on model performance"""
    batch_sizes = [8, 16, 32, 64, 128]
    
    # Load CNN batch size experiment data
    cnn_batch_results = {}
    for bs in batch_sizes:
        history_file = os.path.join(cnn_dir, f'batch_size_{bs}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                cnn_batch_results[bs] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1]
                }
    
    # Load ResNet batch size experiment data
    resnet_batch_results = {}
    for bs in batch_sizes:
        history_file = os.path.join(resnet_dir, f'resnet_batch_size_{bs}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                resnet_batch_results[bs] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1]
                }
    
    # If enough data is available, generate batch size comparison chart
    if cnn_batch_results and resnet_batch_results:
        plt.figure(figsize=(12, 6))
        
        # Accuracy comparison
        cnn_bs = list(cnn_batch_results.keys())
        cnn_acc = [cnn_batch_results[bs]['val_accuracy'] for bs in cnn_bs]
        
        resnet_bs = list(resnet_batch_results.keys())
        resnet_acc = [resnet_batch_results[bs]['val_accuracy'] for bs in resnet_bs]
        
        plt.plot(cnn_bs, cnn_acc, 'o-', label='CNN', linewidth=2, markersize=8, color='#3498db')
        plt.plot(resnet_bs, resnet_acc, 's-', label='ResNet', linewidth=2, markersize=8, color='#e74c3c')
        
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Effect of Batch Size on Validation Accuracy', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xscale('log', base=2)  # Use log scale for better batch size display
        plt.xticks(batch_sizes, batch_sizes)  # Show all batch sizes
        
        # Add value labels to points
        for bs, acc in zip(cnn_bs, cnn_acc):
            plt.text(bs, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
        for bs, acc in zip(resnet_bs, resnet_acc):
            plt.text(bs, acc - 0.01, f'{acc:.4f}', ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_size_comparison.png'), dpi=300)
        plt.close()
        
        print("Batch size comparison chart saved")

def analyze_learning_rates(cnn_dir, resnet_dir, output_dir):
    """Analyze learning rate effects on model performance"""
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    
    # Load CNN learning rate experiment data
    cnn_lr_results = {}
    for lr in learning_rates:
        history_file = os.path.join(cnn_dir, f'learning_rate_{lr}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                cnn_lr_results[lr] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1],
                    'accuracy_curve': history['accuracy'],
                    'val_accuracy_curve': history['val_accuracy']
                }
    
    # Load ResNet learning rate experiment data
    resnet_lr_results = {}
    for lr in learning_rates:
        history_file = os.path.join(resnet_dir, f'resnet_learning_rate_{lr}_history.json')
        if os.path.exists(history_file):
            history = load_json_file(history_file)
            if history:
                resnet_lr_results[lr] = {
                    'val_accuracy': history['val_accuracy'][-1],
                    'val_loss': history['val_loss'][-1],
                    'accuracy_curve': history['accuracy'],
                    'val_accuracy_curve': history['val_accuracy']
                }
    
    # If enough data is available, generate learning rate comparison chart
    if cnn_lr_results and resnet_lr_results:
        # 1. Final performance comparison chart
        plt.figure(figsize=(15, 10))
        
        # Accuracy comparison
        plt.subplot(2, 1, 1)
        
        x = np.arange(len(learning_rates))
        width = 0.35
        
        cnn_acc = [cnn_lr_results.get(lr, {}).get('val_accuracy', 0) for lr in learning_rates]
        resnet_acc = [resnet_lr_results.get(lr, {}).get('val_accuracy', 0) for lr in learning_rates]
        
        plt.bar(x - width/2, cnn_acc, width, label='CNN', color='#3498db')
        plt.bar(x + width/2, resnet_acc, width, label='ResNet', color='#e74c3c')
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy with Different Learning Rates', fontsize=14)
        plt.xticks(x, learning_rates)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, v in enumerate(cnn_acc):
            plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        for i, v in enumerate(resnet_acc):
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        # Loss comparison
        plt.subplot(2, 1, 2)
        
        cnn_loss = [cnn_lr_results.get(lr, {}).get('val_loss', 0) for lr in learning_rates]
        resnet_loss = [resnet_lr_results.get(lr, {}).get('val_loss', 0) for lr in learning_rates]
        
        plt.bar(x - width/2, cnn_loss, width, label='CNN', color='#3498db')
        plt.bar(x + width/2, resnet_loss, width, label='ResNet', color='#e74c3c')
        
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss with Different Learning Rates', fontsize=14)
        plt.xticks(x, learning_rates)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels to bars
        for i, v in enumerate(cnn_loss):
            plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        for i, v in enumerate(resnet_loss):
            plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
        
        plt.suptitle('Effect of Learning Rates on Model Performance', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'learning_rate_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Learning curve comparison for learning rate 0.1 only
        lr = 0.1  # Only show learning rate 0.1
        
        if lr in cnn_lr_results and lr in resnet_lr_results:
            plt.figure(figsize=(15, 6))
            
            # Learning curves
            plt.subplot(1, 2, 1)
            epochs = range(1, len(cnn_lr_results[lr]['accuracy_curve']) + 1)
            plt.plot(epochs, cnn_lr_results[lr]['accuracy_curve'], label='CNN Training Accuracy', linewidth=2, color='#3498db')
            plt.plot(epochs, cnn_lr_results[lr]['val_accuracy_curve'], label='CNN Validation Accuracy', linewidth=2, linestyle='--', color='#3498db')
            plt.plot(epochs, resnet_lr_results[lr]['accuracy_curve'][:len(epochs)], label='ResNet Training Accuracy', linewidth=2, color='#e74c3c')
            plt.plot(epochs, resnet_lr_results[lr]['val_accuracy_curve'][:len(epochs)], label='ResNet Validation Accuracy', linewidth=2, linestyle='--', color='#e74c3c')
            
            plt.title(f'Learning Curves with Learning Rate {lr}', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(epochs)
            
            # Add final accuracy annotations
            plt.annotate(f'CNN Final: {cnn_lr_results[lr]["val_accuracy_curve"][-1]:.4f}',
                        xy=(epochs[-1], cnn_lr_results[lr]["val_accuracy_curve"][-1]),
                        xytext=(epochs[-1]-3, cnn_lr_results[lr]["val_accuracy_curve"][-1]+0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=10)
            
            plt.annotate(f'ResNet Final: {resnet_lr_results[lr]["val_accuracy_curve"][-1]:.4f}',
                        xy=(epochs[-1], resnet_lr_results[lr]["val_accuracy_curve"][-1]),
                        xytext=(epochs[-1]-3, resnet_lr_results[lr]["val_accuracy_curve"][-1]-0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=10)
            
            # Convergence speed comparison
            plt.subplot(1, 2, 2)
            
            # Calculate epochs needed to reach 90% of final accuracy
            cnn_final_acc = cnn_lr_results[lr]['val_accuracy_curve'][-1]
            cnn_threshold = 0.9 * cnn_final_acc
            cnn_convergence = next((i+1 for i, acc in enumerate(cnn_lr_results[lr]['val_accuracy_curve']) if acc >= cnn_threshold), len(epochs))
            
            resnet_final_acc = resnet_lr_results[lr]['val_accuracy_curve'][-1]
            resnet_threshold = 0.9 * resnet_final_acc
            resnet_convergence = next((i+1 for i, acc in enumerate(resnet_lr_results[lr]['val_accuracy_curve']) if acc >= resnet_threshold), len(epochs))
            
            models = ['CNN', 'ResNet']
            convergence_epochs = [cnn_convergence, resnet_convergence]
            
            bars = plt.bar(models, convergence_epochs, color=['#3498db', '#e74c3c'])
            plt.title(f'Convergence Speed with Learning Rate {lr}\n(Epochs to 90% Final Accuracy)', fontsize=14)
            plt.ylabel('Epochs', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels to bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f} epochs',
                        ha='center', va='bottom', fontsize=12)
            
            plt.suptitle('Learning Curves and Convergence Speed Comparison (LR=0.1)', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, 'learning_curves_comparison.png'), dpi=300)
            plt.close()
        
        print("Learning rate comparison charts saved")

def generate_hyperparameter_summary(cnn_dir, resnet_dir, output_dir):
    """Generate hyperparameter optimization summary table"""
    # Load hardware info files to get selected hyperparameters
    try:
        cnn_hw_info_path = os.path.join(cnn_dir, 'hardware_info.json')
        resnet_hw_info_path = os.path.join(resnet_dir, 'resnet_hardware_info.json')
        
        with open(cnn_hw_info_path, 'r') as f:
            cnn_hw_info = json.load(f)
        
        with open(resnet_hw_info_path, 'r') as f:
            resnet_hw_info = json.load(f)
        
        # Create hyperparameter summary table
        plt.figure(figsize=(12, 6))
        
        # Hide axes
        plt.axis('off')
        
        # Create table data
        table_data = [
            ['Model', 'Best Loss Function', 'Best Batch Size', 'Best Learning Rate', 'Validation Accuracy', 'Time/Epoch'],
            ['CNN', cnn_hw_info['selected_loss'], str(cnn_hw_info['selected_batch_size']), 
             str(cnn_hw_info['selected_learning_rate']), '-', f"{cnn_hw_info['final_model_avg_epoch_time']:.2f}s"],
            ['ResNet', resnet_hw_info['selected_loss'], str(resnet_hw_info['selected_batch_size']), 
             str(resnet_hw_info['selected_learning_rate']), '-', f"{resnet_hw_info['final_model_avg_epoch_time']:.2f}s"]
        ]
        
        # Try to load final accuracies
        try:
            cnn_history = load_json_file(os.path.join(cnn_dir, 'final_model_history.json'))
            if cnn_history:
                table_data[1][4] = f"{cnn_history['val_accuracy'][-1]:.4f}"
            
            resnet_history = load_json_file(os.path.join(resnet_dir, 'resnet_final_model_history.json'))
            if resnet_history:
                table_data[2][4] = f"{resnet_history['val_accuracy'][-1]:.4f}"
        except:
            pass
        
        # Create table
        table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.25, 0.15, 0.15, 0.15, 0.15])
        
        # Set table style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Style model rows
        table[(1, 0)].set_facecolor('#3498db')
        table[(1, 0)].set_text_props(color='white', fontweight='bold')
        table[(2, 0)].set_facecolor('#e74c3c')
        table[(2, 0)].set_text_props(color='white', fontweight='bold')
        
        plt.title('CNN vs ResNet Hyperparameter Optimization Summary', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hyperparameter_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Hyperparameter optimization summary table saved")
    except Exception as e:
        print(f"Could not generate hyperparameter summary table: {e}")

def main():
    """Main function to handle command line arguments and execute comparison"""
    parser = argparse.ArgumentParser(description='Compare CNN and ResNet model performance')
    parser.add_argument('--cnn', default='benchmarks/final_model_history.json',
                        help='CNN model history file path (default: benchmarks/final_model_history.json)')
    parser.add_argument('--resnet', default='ResNet/resnet_final_model_history.json',
                        help='ResNet model history file path (default: ResNet/resnet_final_model_history.json)')
    parser.add_argument('--output', default='model_comparison.png',
                        help='Output chart filename (default: model_comparison.png)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: current directory)')
    parser.add_argument('--full-analysis', action='store_true',
                        help='Perform full hyperparameter analysis (default: False)')
    
    args = parser.parse_args()
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Build full file paths
    cnn_history_file = os.path.join(project_root, args.cnn)
    resnet_history_file = os.path.join(project_root, args.resnet)
    
    # Check if files exist
    if not os.path.exists(cnn_history_file):
        print(f"Error: CNN history file does not exist: {cnn_history_file}")
        print("Please run benchmarks/kuzushiji_mnist_cnn.py first to train the CNN model")
        return
    
    if not os.path.exists(resnet_history_file):
        print(f"Error: ResNet history file does not exist: {resnet_history_file}")
        print("Please run ResNet/kuzushiji_mnist_resnet.py first to train the ResNet model")
        return
    
    # If no output directory specified, use project root
    if args.output_dir is None:
        args.output_dir = project_root
    
    # Execute comparison
    print("Generating model comparison charts...")
    compare_models(cnn_history_file, resnet_history_file, args.output, args.output_dir)
    
    if args.full_analysis:
        print("\nPerforming full hyperparameter analysis...")
        cnn_dir = os.path.dirname(cnn_history_file)
        resnet_dir = os.path.dirname(resnet_history_file)
        presentation_dir = os.path.join(args.output_dir, 'presentation')
        
        analyze_hyperparameters(cnn_dir, resnet_dir, presentation_dir)
        generate_hyperparameter_summary(cnn_dir, resnet_dir, presentation_dir)
    
    print("\nAll charts generated successfully! Saved in:", os.path.join(args.output_dir, 'presentation'))
    print("Please check the 'presentation' directory for chart files")

if __name__ == "__main__":
    main()