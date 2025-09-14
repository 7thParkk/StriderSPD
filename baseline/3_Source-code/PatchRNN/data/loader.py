"""
DataLoader for loading security patch data
"""
import json
import os
from typing import List, Any

class DataLoader:
    """Simple data loader for JSON files"""
    
    def __init__(self, config):
        self.config = config
        # Use mydata path where the actual JSON files are located
        self.data_path = "./mydata/"
    
    def read_all_splits(self) -> tuple:
        """Read train, validation, and test data splits"""
        try:
            train_data = self._read_json(os.path.join(self.data_path, 'train.json'))
            valid_data = self._read_json(os.path.join(self.data_path, 'valid.json')) 
            test_data = self._read_json(os.path.join(self.data_path, 'test.json'))
            return train_data, valid_data, test_data
        except Exception as e:
            print(f"[ERROR] Failed to read data splits: {e}")
            # Return empty lists as fallback
            return [], [], []
    
    def _read_json(self, file_path: str) -> List[Any]:
        """Read JSON file"""
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            return []