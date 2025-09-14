"""
Training module for Security Patch Identification RNN
Maintains compatibility with original training logic
"""
import os
import pickle
import numpy as np
import torch
from typing import Tuple, Any

from config import Config
from data.loader import DataLoader
from data.processor import DataProcessor
from models.diff_rnn import DiffRNN, DiffRNNTrainer


class Trainer:
    """Main trainer class maintaining original functionality"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)
        # Initialize DiffRNN trainer
        self.diff_trainer = DiffRNNTrainer(config)
        self.device = config.device
        
    def demo_diff_rnn(self) -> Tuple[Any, float]:
        """
        Demo program for DiffRNN (maintaining original demoDiffRNN logic)
        
        Returns:
            Tuple of (model, accuracy)
        """
        print("[INFO] Starting DiffRNN demo...")
        
        # Load pre-split data from JSON files
        try:
            train_raw_data, valid_raw_data, test_raw_data = self.data_loader.read_all_splits()
            all_raw_data = train_raw_data + valid_raw_data + test_raw_data
            
            # Cache all data for processing
            data_cache_path = os.path.join(self.config.temp_path, 'data.npy')
            np.save(data_cache_path, np.array(all_raw_data, dtype=object))
            print(f'[INFO] <ReadData> Saved {len(all_raw_data)} raw data to {data_cache_path}')
            
        except Exception as e:
            print(f"[WARNING] Failed to load JSON data: {e}")
            # Fall back to cached data if available
            data_cache_path = os.path.join(self.config.temp_path, 'data.npy')
            if os.path.exists(data_cache_path):
                all_raw_data = np.load(data_cache_path, allow_pickle=True).tolist()
                print(f'[INFO] <ReadData> Load {len(all_raw_data)} raw data from {data_cache_path}')
            else:
                raise Exception("No data available")
        
        # Get diff file properties for all data
        props_cache_path = os.path.join(self.config.temp_path, 'props.pkl')
        if not os.path.exists(props_cache_path):
            all_diff_props = self.data_processor.get_diff_props(all_raw_data)
        else:
            with open(props_cache_path, 'rb') as f:
                all_diff_props = pickle.load(f)
            print(f'[INFO] <GetDiffProps> Load {len(all_diff_props)} diff property data from {props_cache_path}')
        
        # Get diff token vocabulary from all data
        diff_vocab, diff_max_len = self.data_processor.get_diff_vocab(all_diff_props)
        
        # Limit max length as in original
        diff_max_len = self.config.diff_max_len if (diff_max_len > self.config.diff_max_len) else diff_max_len
        
        # Get diff token dictionary
        diff_dict = self.data_processor.get_diff_dict(diff_vocab)
        
        # Get pre-trained weights for embedding layer
        diff_pre_weights = self.data_processor.get_diff_embed(diff_dict, self.config.diff_embed_dim)
        
        # Since we have pre-processed props for all data, we need to split them correctly
        # The props correspond to the order: train + valid + test
        train_size = len(train_raw_data)
        valid_size = len(valid_raw_data)
        test_size = len(test_raw_data)
        
        # Split the props data according to the original data splits
        train_props = all_diff_props[:train_size]
        valid_props = all_diff_props[train_size:train_size + valid_size]
        test_props = all_diff_props[train_size + valid_size:train_size + valid_size + test_size]
        
        print(f"[INFO] Processing {len(train_props)} train samples...")
        data_train, label_train = self.data_processor.get_diff_mapping(train_props, diff_max_len, diff_dict)
        data_train = self.data_processor.update_token_types(data_train)
        
        print(f"[INFO] Processing {len(valid_props)} valid samples...")
        data_valid, label_valid = self.data_processor.get_diff_mapping(valid_props, diff_max_len, diff_dict)
        data_valid = self.data_processor.update_token_types(data_valid)
        
        print(f"[INFO] Processing {len(test_props)} test samples...")
        data_test, label_test = self.data_processor.get_diff_mapping(test_props, diff_max_len, diff_dict)
        data_test = self.data_processor.update_token_types(data_test)
        
        print(f'[INFO] <main> Get {len(data_train)} TRAIN data, {len(data_valid)} VALID data, '
              f'{len(data_test)} TEST data. (Total: {len(data_train)+len(data_valid)+len(data_test)})')
        
        # Train or load DiffRNN model
        model_path = os.path.join(self.config.temp_path, 'model_DiffRNN.pth')
        if self.config.load_model and os.path.exists(model_path):
            # Load pre-trained model
            pre_weights = torch.from_numpy(diff_pre_weights)
            model = DiffRNN(
                pre_weights=pre_weights,
                hidden_size=self.config.diff_rnn_hidden_size,
                hidden_layers=self.config.diff_rnn_hidden_layers
            )
            model.load_state_dict(torch.load(model_path))
            print(f'[INFO] Loaded pre-trained DiffRNN model from {model_path}')
        else:
            # Train new model
            model = self.diff_trainer.train(
                data_train, label_train, data_valid, label_valid, 
                diff_pre_weights, data_test, label_test
            )
        
        # Test DiffRNN
        predictions, accuracy = self.diff_trainer.test(model, data_test, label_test)
        
        # Output evaluation
        confusion, report = self._output_eval(predictions, label_test, 'DiffRNN')
        
        return model, accuracy
    
    def demo_msg_rnn(self) -> Tuple[Any, float]:
        """
        Demo program for MsgRNN
        
        Returns:
            Tuple of (model, accuracy)
        """
        # Implementation would go here
        pass
    
    def demo_patch_rnn(self) -> Tuple[Any, float]:
        """
        Demo program for PatchRNN
        
        Returns:
            Tuple of (model, accuracy)
        """
        # Implementation would go here
        pass
    
    def demo_twin_rnn(self) -> Tuple[Any, float]:
        """
        Demo program for TwinRNN
        
        Returns:
            Tuple of (model, accuracy)
        """
        # Implementation would go here
        pass
    
    def _output_eval(self, predictions, labels, model_name):
        """
        Output evaluation metrics
        
        Args:
            predictions: Model predictions
            labels: True labels
            model_name: Name of the model
            
        Returns:
            Tuple of (confusion_matrix, classification_report)
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        confusion = confusion_matrix(labels, predictions)
        report = classification_report(labels, predictions)
        
        print(f"\n[INFO] {model_name} Evaluation Results:")
        print("Confusion Matrix:")
        print(confusion)
        print("\nClassification Report:")
        print(report)
        
        return confusion, report


def main():
    """Main function to run the training pipeline"""
    print("Starting PatchRNN training pipeline...")
    
    # Initialize configuration
    config = Config()
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Run DiffRNN demo
    print("Running DiffRNN training...")
    try:
        model, accuracy = trainer.demo_diff_rnn()
        print(f"[INFO] DiffRNN training completed with accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"[ERROR] DiffRNN training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()