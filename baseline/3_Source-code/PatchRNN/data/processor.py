"""
DataProcessor for processing security patch data
"""
from typing import List, Dict, Any, Tuple
import re

class DataProcessor:
    """Simple data processor for diff data"""
    
    def __init__(self, config):
        self.config = config
    
    def get_diff_props(self, raw_data: List[Any]) -> List[Dict[str, Any]]:
        """Extract diff properties from raw data"""
        diff_props = []
        
        for item in raw_data:
            if isinstance(item, dict):
                # Extract basic properties
                props = {
                    'diff': item.get('diff', ''),
                    'msg': item.get('msg', ''),
                    'label': item.get('label', 0),
                    'commit_id': item.get('commit_id', '')
                }
                diff_props.append(props)
        
        print(f"[INFO] Processed {len(diff_props)} diff properties")
        return diff_props
    
    def get_diff_vocab(self, diff_props: List[Dict[str, Any]]) -> Tuple[Dict[str, int], int]:
        """Create vocabulary from diff data"""
        vocab = {}
        max_len = 0
        
        for props in diff_props:
            diff_text = props.get('diff', '')
            if diff_text:
                # Simple tokenization - split by whitespace and special characters
                tokens = re.findall(r'\w+|[^\w\s]', diff_text)
                max_len = max(max_len, len(tokens))
                
                for token in tokens:
                    if token not in vocab:
                        vocab[token] = len(vocab) + 1  # 0 reserved for padding
        
        print(f"[INFO] Created vocabulary with {len(vocab)} tokens, max sequence length: {max_len}")
        return vocab, max_len
    
    def get_diff_dict(self, diff_vocab: Dict[str, int]) -> Dict[str, int]:
        """Convert vocabulary to dictionary format"""
        # Add special tokens
        diff_dict = {'<PAD>': 0, '<UNK>': 1}
        
        # Add vocabulary tokens
        for token, idx in diff_vocab.items():
            diff_dict[token] = idx + 2  # Offset by 2 to account for special tokens
        
        print(f"[INFO] Created diff dictionary with {len(diff_dict)} tokens")
        return diff_dict
    
    def get_diff_embed(self, diff_dict: Dict[str, int], embed_dim: int) -> Any:
        """Create embedding weights for diff tokens"""
        import torch
        
        vocab_size = len(diff_dict)
        # Create random embedding weights (in a real implementation, this would use pre-trained embeddings)
        embedding_weights = torch.randn(vocab_size, embed_dim)
        
        print(f"[INFO] Created embedding weights with shape {embedding_weights.shape}")
        return embedding_weights
    
    def get_diff_mapping(self, props: List[Dict[str, Any]], max_len: int, diff_dict: Dict[str, int]) -> tuple:
        """Convert diff properties to numerical mappings"""
        import numpy as np
        
        data = []
        labels = []
        
        for prop in props:
            diff_text = prop.get('diff', '')
            label = prop.get('label', 0)
            
            # Tokenize and convert to numerical indices
            tokens = self._tokenize_diff(diff_text)
            token_indices = [diff_dict.get(token, diff_dict['<UNK>']) for token in tokens]
            
            # Pad or truncate to max_len
            if len(token_indices) > max_len:
                token_indices = token_indices[:max_len]
            else:
                token_indices = token_indices + [diff_dict['<PAD>']] * (max_len - len(token_indices))
            
            data.append(token_indices)
            labels.append(label)
        
        return np.array(data), np.array(labels)
    
    def update_token_types(self, data: Any) -> Any:
        """Update token types (placeholder implementation)"""
        # In a real implementation, this would modify token types
        # For now, just return the data as-is
        return data
    
    def _tokenize_diff(self, diff_text: str) -> List[str]:
        """Tokenize diff text"""
        import re
        if not diff_text:
            return []
        # Simple tokenization - split by whitespace and special characters
        return re.findall(r'\w+|[^\w\s]', diff_text)