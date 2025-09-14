"""
Main entry point for Security Patch Identification RNN
Maintains compatibility with original SecurityPatchIdentificationRNN.py
"""
import os
import sys
import argparse
import torch
import numpy as np
import random
from typing import Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import get_config
from training.trainer import Trainer
from training.evaluator import Evaluator

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main function maintaining original workflow"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Security Patch Identification RNN')
    parser.add_argument('--model', type=str, default='diff', 
                       choices=['diff', 'msg', 'patch', 'twin', 'all'],
                       help='Model to train/test (default: diff)')
    parser.add_argument('--load-model', action='store_true',
                       help='Load pre-trained model instead of training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Get configuration
    config = get_config()
    
    # Update config based on arguments
    config.load_model = args.load_model
    config.debug_mode = args.debug
    
    if args.device != 'auto':
        config.device = args.device
    
    # Print configuration
    print("=" * 80)
    print("SECURITY PATCH IDENTIFICATION RNN")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Device: {config.device}")
    print(f"Load Model: {config.load_model}")
    print(f"Debug Mode: {config.debug_mode}")
    print(f"Random Seed: {args.seed}")
    print("=" * 80)
    
    # Initialize trainer and evaluator
    trainer = Trainer(config)
    evaluator = Evaluator(config)
    
    # Store results for comparison
    all_results = []
    
    try:
        if args.model == 'diff' or args.model == 'all':
            print("\n[INFO] Training/Testing DiffRNN...")
            diff_model, diff_accuracy = trainer.demo_diff_rnn()
            print(f"[INFO] DiffRNN Accuracy: {diff_accuracy:.4f}")
            
            if args.model == 'all':
                # Store for later use in combined models
                globals()['diff_model'] = diff_model
        
        if args.model == 'msg' or args.model == 'all':
            print("\n[INFO] Training/Testing MsgRNN...")
            msg_model, msg_accuracy = trainer.demo_msg_rnn()
            print(f"[INFO] MsgRNN Accuracy: {msg_accuracy:.4f}")
            
            if args.model == 'all':
                # Store for later use in combined models
                globals()['msg_model'] = msg_model
        
        if args.model == 'patch' or args.model == 'all':
            print("\n[INFO] Training/Testing PatchRNN...")
            if args.model == 'all' and 'diff_model' in globals() and 'msg_model' in globals():
                patch_model, patch_accuracy = trainer.demo_patch_rnn(
                    globals()['diff_model'], globals()['msg_model']
                )
                print(f"[INFO] PatchRNN Accuracy: {patch_accuracy:.4f}")
            else:
                print("[WARNING] PatchRNN requires both DiffRNN and MsgRNN models. Run with --model all")
        
        if args.model == 'twin' or args.model == 'all':
            print("\n[INFO] Training/Testing TwinRNN...")
            twin_model, twin_accuracy = trainer.demo_twin_rnn()
            print(f"[INFO] TwinRNN Accuracy: {twin_accuracy:.4f}")
        
        print("\n[INFO] Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def demo_diff_rnn():
    """
    Standalone demo function for DiffRNN (maintaining original interface)
    """
    config = get_config()
    trainer = Trainer(config)
    return trainer.demo_diff_rnn()

def demo_msg_rnn():
    """
    Standalone demo function for MsgRNN
    """
    config = get_config()
    trainer = Trainer(config)
    return trainer.demo_msg_rnn()

def demo_patch_rnn(diff_model, msg_model):
    """
    Standalone demo function for PatchRNN
    """
    config = get_config()
    trainer = Trainer(config)
    return trainer.demo_patch_rnn(diff_model, msg_model)

def demo_twin_rnn():
    """
    Standalone demo function for TwinRNN
    """
    config = get_config()
    trainer = Trainer(config)
    return trainer.demo_twin_rnn()

# Maintain compatibility with original function names
demoDiffRNN = demo_diff_rnn
demoMsgRNN = demo_msg_rnn
demoPatchRNN = demo_patch_rnn
demoTwinRNN = demo_twin_rnn

if __name__ == "__main__":
    main()