"""
MsgRNN: Commit Message Analysis Network
Maintains exact compatibility with original architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import gc
import os
import datetime
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

class MsgRNN(nn.Module):
    """
    MsgRNN: Convert commit message data into predicted labels
    Maintains exact architecture compatibility with original implementation
    """
    
    def __init__(self, config, pre_weights: torch.Tensor = None, hidden_size: int = None, hidden_layers: int = None):
        """
        Define each layer in the network model for message processing
        
        Args:
            pre_weights: Pre-trained weights for embedding layer
            hidden_size: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
        """
        super(MsgRNN, self).__init__()
        
        # Parameters from config
        self.num_classes = 2
        hidden_size = hidden_size or config.msg_rnn_hidden_size
        hidden_layers = hidden_layers or config.msg_rnn_hidden_layers
        
        # Initialize embedding layer with pre-trained weights if available
        if pre_weights is not None:
            vocab_size, embed_dim = pre_weights.size()
        elif hasattr(config, 'pre_weights') and config.pre_weights is not None:
            vocab_size, embed_dim = config.pre_weights.size()
            pre_weights = config.pre_weights
        else:
            # Use default vocabulary size and embedding dimension
            vocab_size = getattr(config, 'vocab_size', 10000)
            embed_dim = config.msg_embed_dim
        
        # Embedding Layer - exactly as original
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        if pre_weights is not None:
            self.embedding.load_state_dict({'weight': pre_weights})
        self.embedding.weight.requires_grad = True
        
        # LSTM Layer - bidirectional as original
        # Message RNN doesn't have extra dimensions like DiffRNN
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            bidirectional=True,
            batch_first=False  # Original uses seq_len first
        )
        
        # Fully-Connected Layer
        self.fc = nn.Linear(hidden_size * hidden_layers * 2, self.num_classes)
        
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=-1)
        
        # Store parameters for reference
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert message inputs to predictions
        
        Args:
            x: Input tensor with shape (batch_size, msg_length)
               containing token IDs
        
        Returns:
            Softmax predictions with shape (batch_size, num_classes)
        """
        # x shape: batch_size * msg_length
        
        # Embed token IDs
        embeds = self.embedding(x)
        # embeds shape: batch_size * msg_length * embedding_dim
        
        # Permute to match LSTM expected input: seq_len * batch_size * input_size
        inputs = embeds.permute(1, 0, 2)
        # inputs shape: msg_length * batch_size * embedding_dim
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(inputs)
        # lstm_out shape: msg_length * batch_size * (hidden_size * 2)
        # h_n shape: (num_layers * 2) * batch_size * hidden_size
        
        # Concatenate all hidden states from final layer
        feature_map = torch.cat([h_n[i, :, :] for i in range(h_n.shape[0])], dim=1)
        # feature_map shape: batch_size * (hidden_size * num_layers * 2)
        
        # Final classification layer
        final_out = self.fc(feature_map)
        # final_out shape: batch_size * num_classes
        
        return self.softmax(final_out)


class MsgRNNTrainer:
    """
    Trainer class for MsgRNN maintaining original training logic
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize logging
        self.log_dir = os.path.join(config.temp_path, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f'msg_rnn_training_{timestamp}.log')
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"MsgRNN Training Log - Started at {datetime.datetime.now()}\n")
            f.write("="*80 + "\n")
            f.write(f"Configuration:\n")
            f.write(f"  Hidden Size: {config.msg_rnn_hidden_size}\n")
            f.write(f"  Hidden Layers: {config.msg_rnn_hidden_layers}\n")
            f.write(f"  Batch Size: {config.msg_rnn_batch_size}\n")
            f.write(f"  Learning Rate: {config.msg_rnn_learn_rate}\n")
            f.write(f"  Max Epochs: {config.msg_rnn_max_epoch}\n")
            f.write(f"  Judge Epoch: {config.msg_rnn_judge_epoch}\n")
            f.write(f"  Device: {config.device}\n")
            f.write("="*80 + "\n\n")

    def calculate_detailed_metrics(self, y_true: List[int], y_pred: List[int], dataset_name: str) -> Dict[str, Any]:
        """Calculate detailed metrics including corrected confusion matrix"""
        
        # Calculate confusion matrix (sklearn format: rows=true, cols=predicted)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Extract values from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics manually for verification
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Use sklearn for verification
        sklearn_precision, sklearn_recall, sklearn_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        sklearn_accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'dataset': dataset_name,
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'matrix': cm.tolist()
            },
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1_score),
                'fpr': float(fpr),
                'fnr': float(fnr)
            },
            'sklearn_verification': {
                'accuracy': float(sklearn_accuracy),
                'precision': float(sklearn_precision),
                'recall': float(sklearn_recall),
                'f1_score': float(sklearn_f1)
            },
            'predictions': y_pred,
            'true_labels': y_true
        }

    def save_epoch_results(self, epoch: int, train_metrics: Dict, valid_metrics: Dict, 
                          test_metrics: Optional[Dict], train_loss: float, 
                          test_data_details: Optional[List[Dict]] = None):
        """Save detailed results for each epoch"""
        
        epoch_data = {
            'epoch': epoch + 1,
            'timestamp': datetime.datetime.now().isoformat(),
            'train_loss': float(train_loss),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'test_metrics': test_metrics
        }
        
        # Save to JSON file
        epoch_file = os.path.join(self.log_dir, f'msg_rnn_epoch_{epoch+1:03d}_results.json')
        with open(epoch_file, 'w') as f:
            json.dump(epoch_data, f, indent=2)
        
        # Save detailed test data predictions if available
        if test_data_details:
            test_details_file = os.path.join(self.log_dir, f'msg_rnn_epoch_{epoch+1:03d}_test_details.json')
            with open(test_details_file, 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'test_data_count': len(test_data_details),
                    'test_predictions': test_data_details
                }, f, indent=2)
        
        # Also append to summary file
        summary_file = os.path.join(self.log_dir, 'msg_rnn_training_summary.jsonl')
        with open(summary_file, 'a') as f:
            json.dump(epoch_data, f)
            f.write('\n')

    def collect_test_data_details(self, model: MsgRNN, test_loader: torchdata.DataLoader, 
                                 epoch: int) -> List[Dict]:
        """Collect detailed predictions for each test data sample"""
        model.eval()
        test_details = []
        
        with torch.no_grad():
            sample_idx = 0
            for batch_idx, (data, labels) in enumerate(test_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Process each sample in the batch
                for i in range(data.size(0)):
                    sample_detail = {
                        'sample_id': sample_idx,
                        'batch_id': batch_idx,
                        'batch_position': i,
                        'true_label': int(labels[i].cpu().item()),
                        'predicted_label': int(predictions[i].cpu().item()),
                        'prediction_probabilities': {
                            'class_0_prob': float(probabilities[i][0].cpu().item()),
                            'class_1_prob': float(probabilities[i][1].cpu().item())
                        },
                        'confidence': float(torch.max(probabilities[i]).cpu().item()),
                        'is_correct': bool(predictions[i] == labels[i]),
                        'prediction_type': self._get_prediction_type(
                            int(labels[i].cpu().item()), 
                            int(predictions[i].cpu().item())
                        ),
                        'input_data_shape': list(data[i].shape),
                        'input_sequence_length': int(torch.sum(data[i] != 0).item())
                    }
                    test_details.append(sample_detail)
                    sample_idx += 1
        
        return test_details
    
    def _get_prediction_type(self, true_label: int, pred_label: int) -> str:
        """Get the type of prediction (TP, TN, FP, FN)"""
        if true_label == 1 and pred_label == 1:
            return 'TP'  # True Positive
        elif true_label == 0 and pred_label == 0:
            return 'TN'  # True Negative
        elif true_label == 0 and pred_label == 1:
            return 'FP'  # False Positive
        elif true_label == 1 and pred_label == 0:
            return 'FN'  # False Negative
        else:
            return 'Unknown'

    def _write_detailed_metrics(self, f, metrics: Dict):
        """Write detailed metrics to log file"""
        cm = metrics['confusion_matrix']
        m = metrics['metrics']
        
        f.write(f"Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"                 Predicted\n")
        f.write(f"                 0(Neg)  1(Pos)\n")
        f.write(f"Actual 0(Neg)    {cm['tn']:5d}   {cm['fp']:5d}\n")
        f.write(f"       1(Pos)    {cm['fn']:5d}   {cm['tp']:5d}\n\n")
        
        f.write(f"Detailed Metrics:\n")
        f.write(f"  Accuracy:    {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:   {m['precision']:.4f} ({m['precision']*100:.2f}%)\n")
        f.write(f"  Recall:      {m['recall']:.4f} ({m['recall']*100:.2f}%)\n")
        f.write(f"  Specificity: {m['specificity']:.4f} ({m['specificity']*100:.2f}%)\n")
        f.write(f"  F1-Score:    {m['f1_score']:.4f}\n")
        f.write(f"  FPR:         {m['fpr']:.4f} ({m['fpr']*100:.2f}%)\n")
        f.write(f"  FNR:         {m['fnr']:.4f} ({m['fnr']*100:.2f}%)\n\n")
        
        # Verification with sklearn
        sk = metrics['sklearn_verification']
        f.write(f"Sklearn Verification:\n")
        f.write(f"  Accuracy:    {sk['accuracy']:.4f}\n")
        f.write(f"  Precision:   {sk['precision']:.4f}\n")
        f.write(f"  Recall:      {sk['recall']:.4f}\n")
        f.write(f"  F1-Score:    {sk['f1_score']:.4f}\n")
        
    def train(self, 
              m_train: np.ndarray, 
              l_train: np.ndarray,
              m_valid: np.ndarray, 
              l_valid: np.ndarray,
              pre_weights: np.ndarray,
              m_test: Optional[np.ndarray] = None,
              l_test: Optional[np.ndarray] = None) -> MsgRNN:
        """
        Train the MsgRNN model with exact original logic
        
        Args:
            m_train: Training message data
            l_train: Training labels  
            m_valid: Validation message data
            l_valid: Validation labels
            pre_weights: Pre-trained embedding weights
            m_test: Optional test message data
            l_test: Optional test labels
            
        Returns:
            Trained MsgRNN model
        """
        # Check if test data is provided
        mark_test = (m_test is not None) and (l_test is not None)
        
        # Convert to tensors and move to device
        x_train = torch.from_numpy(m_train).long().to(self.device)
        y_train = torch.from_numpy(l_train).long().to(self.device)
        x_valid = torch.from_numpy(m_valid).long().to(self.device)
        y_valid = torch.from_numpy(l_valid).long().to(self.device)
        
        if mark_test:
            x_test = torch.from_numpy(m_test).long().to(self.device)
            y_test = torch.from_numpy(l_test).long().to(self.device)
        
        # Create data loaders
        train_dataset = torchdata.TensorDataset(x_train, y_train)
        train_loader = torchdata.DataLoader(
            train_dataset, 
            batch_size=self.config.msg_rnn_batch_size, 
            shuffle=True
        )
        
        valid_dataset = torchdata.TensorDataset(x_valid, y_valid)
        valid_loader = torchdata.DataLoader(
            valid_dataset, 
            batch_size=self.config.msg_rnn_batch_size, 
            shuffle=False
        )
        
        if mark_test:
            test_dataset = torchdata.TensorDataset(x_test, y_test)
            test_loader = torchdata.DataLoader(
                test_dataset, 
                batch_size=self.config.msg_rnn_batch_size, 
                shuffle=False
            )
        
        # Calculate class weights as in original
        l_train_flat = l_train.flatten()
        weights = []
        for lb in range(2):
            weights.append(1 - np.sum(l_train_flat == lb) / len(l_train_flat))
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        # Build model
        pre_weights_tensor = torch.from_numpy(pre_weights)
        model = MsgRNN(
            self.config,
            pre_weights=pre_weights_tensor
        )
        model.to(self.device)
        
        print(f'[INFO] <MsgRNNTrain> ModelType: MsgRNN, HiddenNodes: {self.config.msg_rnn_hidden_size}, HiddenLayers: {self.config.msg_rnn_hidden_layers}')
        print(f'[INFO] <MsgRNNTrain> BatchSize: {self.config.msg_rnn_batch_size}, LearningRate: {self.config.msg_rnn_learn_rate:.4f}, MaxEpoch: {self.config.msg_rnn_max_epoch}')
        
        # Optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=self.config.msg_rnn_learn_rate)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_valid_acc = 0.0
        acc_list = [0]
        
        for epoch in range(self.config.msg_rnn_max_epoch):
            # Training phase
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_labels = []
            
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.contiguous().view(-1).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * len(labels)
                preds = outputs.max(1)[1]
                train_predictions.extend(preds.cpu().tolist())
                train_labels.extend(labels.cpu().tolist())
                
                # Memory cleanup
                torch.cuda.empty_cache()
            
            # Calculate training accuracy
            train_acc = accuracy_score(train_labels, train_predictions) * 100
            train_loss = train_loss / len(train_labels)
            
            # Validation phase
            model.eval()
            valid_predictions = []
            valid_labels = []
            
            with torch.no_grad():
                for data, labels in valid_loader:
                    data = data.to(self.device)
                    labels = labels.contiguous().view(-1).to(self.device)
                    
                    outputs = model(data)
                    preds = outputs.max(1)[1]
                    
                    valid_predictions.extend(preds.cpu().tolist())
                    valid_labels.extend(labels.cpu().tolist())
            
            valid_acc = accuracy_score(valid_labels, valid_predictions) * 100
            
            # Test phase (if test data provided)
            test_acc = None
            if mark_test:
                model.eval()
                test_predictions = []
                test_labels = []
                
                with torch.no_grad():
                    for data, labels in test_loader:
                        data = data.to(self.device)
                        labels = labels.contiguous().view(-1).to(self.device)
                        
                        outputs = model(data)
                        preds = outputs.max(1)[1]
                        
                        test_predictions.extend(preds.cpu().tolist())
                        test_labels.extend(labels.cpu().tolist())
                
                test_acc = accuracy_score(test_labels, test_predictions) * 100
            
            # Logging
            if epoch % self.config.msg_rnn_per_epoch == 0:
                log_str = f'[MsgRNN Epoch {epoch+1:03}/{self.config.msg_rnn_max_epoch:03}] Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Valid: {valid_acc:.2f}%'
                if mark_test and test_acc is not None:
                    log_str += f' | Test: {test_acc:.2f}%'
                print(log_str)
                
                # Detailed logging
                train_metrics = self.calculate_detailed_metrics(train_labels, train_predictions, 'train')
                valid_metrics = self.calculate_detailed_metrics(valid_labels, valid_predictions, 'valid')
                test_metrics = None
                if mark_test and test_acc is not None:
                    test_metrics = self.calculate_detailed_metrics(test_labels, test_predictions, 'test')
                
                # Save epoch results
                self.save_epoch_results(epoch, train_metrics, valid_metrics, test_metrics, train_loss)
                
                # Write to log file
                with open(self.log_file, 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"EPOCH {epoch+1:03d} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Loss: {train_loss:.6f}\n")
                    f.write(f"Train Accuracy: {train_acc:.2f}%\n")
                    f.write(f"Valid Accuracy: {valid_acc:.2f}%\n")
                    if test_acc is not None:
                        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
                    f.write(f"Train-Valid Gap: {train_acc - valid_acc:.2f}%\n\n")
                    
                    f.write("TRAINING SET METRICS:\n")
                    f.write("-" * 50 + "\n")
                    self._write_detailed_metrics(f, train_metrics)
                    
                    f.write("\nVALIDATION SET METRICS:\n")
                    f.write("-" * 50 + "\n")
                    self._write_detailed_metrics(f, valid_metrics)
                    
                    if test_metrics is not None:
                        f.write("\nTEST SET METRICS:\n")
                        f.write("-" * 50 + "\n")
                        self._write_detailed_metrics(f, test_metrics)
                    
                    f.write(f"\n{'='*80}\n")
            
            # Early stopping check
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                # Save best model
                torch.save(model.state_dict(), f'{self.config.temp_path}/model_MsgRNN_best.pth')
            
            acc_list.append(valid_acc)
            
            # Early stopping condition
            if epoch >= self.config.msg_rnn_judge_epoch:
                recent_accs = acc_list[-self.config.msg_rnn_judge_epoch:]
                if all(acc <= best_valid_acc for acc in recent_accs):
                    print(f'[INFO] Early stopping at epoch {epoch+1}')
                    break
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load best model
        model.load_state_dict(torch.load(f'{self.config.temp_path}/model_MsgRNN_best.pth'))
        
        # Save final model
        torch.save(model.state_dict(), f'{self.config.temp_path}/model_MsgRNN.pth')
        
        return model
    
    def test(self, model: MsgRNN, m_test: np.ndarray, l_test: np.ndarray) -> Tuple[List[int], float]:
        """
        Test the MsgRNN model
        
        Args:
            model: Trained MsgRNN model
            m_test: Test message data
            l_test: Test labels
            
        Returns:
            Tuple of (predictions, accuracy)
        """
        model.eval()
        
        x_test = torch.from_numpy(m_test).long().to(self.device)
        y_test = torch.from_numpy(l_test).long().to(self.device)
        
        test_dataset = torchdata.TensorDataset(x_test, y_test)
        test_loader = torchdata.DataLoader(
            test_dataset, 
            batch_size=self.config.msg_rnn_batch_size, 
            shuffle=False
        )
        
        predictions = []
        labels = []
        
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(self.device)
                label = label.contiguous().view(-1).to(self.device)
                
                outputs = model(data)
                preds = outputs.max(1)[1]
                
                predictions.extend(preds.cpu().tolist())
                labels.extend(label.cpu().tolist())
        
        accuracy = accuracy_score(labels, predictions) * 100
        
        return predictions, accuracy