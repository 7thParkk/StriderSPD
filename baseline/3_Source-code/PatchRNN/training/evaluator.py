"""
Evaluation utilities for Security Patch Identification RNN
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import List, Dict, Any, Tuple
import os

class Evaluator:
    """
    Comprehensive evaluation utilities
    """
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_model(self, predictions: List[int], labels: List[int], model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            predictions: Model predictions
            labels: True labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(labels, predictions, 
                                     target_names=['Non-Security', 'Security'],
                                     output_dict=True)
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Security', 'Security'],
                   yticklabels=['Non-Security', 'Security'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.config.temp_path, f'confusion_matrix_{model_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"[INFO] Confusion matrix saved to {save_path}")
    
    def plot_metrics_comparison(self, results_list: List[Dict[str, Any]], save_path: str = None):
        """
        Plot comparison of metrics across different models
        
        Args:
            results_list: List of evaluation results from different models
            save_path: Path to save the plot
        """
        models = [r['model_name'] for r in results_list]
        accuracies = [r['accuracy'] for r in results_list]
        precisions = [r['precision'] for r in results_list]
        recalls = [r['recall'] for r in results_list]
        f1_scores = [r['f1_score'] for r in results_list]
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
            ax.text(i - 1.5*width, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            ax.text(i - 0.5*width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
            ax.text(i + 0.5*width, rec + 0.01, f'{rec:.3f}', ha='center', va='bottom')
            ax.text(i + 1.5*width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.config.temp_path, 'metrics_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"[INFO] Metrics comparison saved to {save_path}")
    
    def generate_report(self, results_list: List[Dict[str, Any]], save_path: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results_list: List of evaluation results from different models
            save_path: Path to save the report
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SECURITY PATCH IDENTIFICATION RNN - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary table
        report_lines.append("SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        report_lines.append("-" * 60)
        
        for result in results_list:
            report_lines.append(
                f"{result['model_name']:<15} "
                f"{result['accuracy']:<10.4f} "
                f"{result['precision']:<10.4f} "
                f"{result['recall']:<10.4f} "
                f"{result['f1_score']:<10.4f}"
            )
        
        report_lines.append("")
        
        # Detailed results for each model
        for result in results_list:
            report_lines.append(f"DETAILED RESULTS - {result['model_name']}")
            report_lines.append("-" * 40)
            
            # Overall metrics
            report_lines.append(f"Overall Accuracy: {result['accuracy']:.4f}")
            report_lines.append(f"Overall Precision: {result['precision']:.4f}")
            report_lines.append(f"Overall Recall: {result['recall']:.4f}")
            report_lines.append(f"Overall F1-Score: {result['f1_score']:.4f}")
            report_lines.append("")
            
            # Per-class metrics
            report_lines.append("Per-Class Metrics:")
            classes = ['Non-Security', 'Security']
            for i, class_name in enumerate(classes):
                if i < len(result['precision_per_class']):
                    report_lines.append(
                        f"  {class_name}: "
                        f"Precision={result['precision_per_class'][i]:.4f}, "
                        f"Recall={result['recall_per_class'][i]:.4f}, "
                        f"F1={result['f1_per_class'][i]:.4f}"
                    )
            report_lines.append("")
            
            # Confusion matrix
            report_lines.append("Confusion Matrix:")
            cm = np.array(result['confusion_matrix'])
            report_lines.append("                Predicted")
            report_lines.append("              Non-Sec  Security")
            report_lines.append(f"Actual Non-Sec   {cm[0,0]:6d}    {cm[0,1]:6d}")
            report_lines.append(f"       Security  {cm[1,0]:6d}    {cm[1,1]:6d}")
            report_lines.append("")
            report_lines.append("-" * 60)
            report_lines.append("")
        
        # Best model summary
        if results_list:
            best_model = max(results_list, key=lambda x: x['f1_score'])
            report_lines.append("BEST MODEL")
            report_lines.append("-" * 40)
            report_lines.append(f"Model: {best_model['model_name']}")
            report_lines.append(f"F1-Score: {best_model['f1_score']:.4f}")
            report_lines.append(f"Accuracy: {best_model['accuracy']:.4f}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        else:
            save_path = os.path.join(self.config.temp_path, 'evaluation_report.txt')
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        print(f"[INFO] Evaluation report saved to {save_path}")
        
        return report_text
    
    def calculate_statistical_significance(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate statistical significance between two models (placeholder)
        
        Args:
            results1: Results from first model
            results2: Results from second model
            
        Returns:
            Dictionary with significance test results
        """
        # This is a placeholder for statistical significance testing
        # In a full implementation, you would use McNemar's test or similar
        
        diff_accuracy = abs(results1['accuracy'] - results2['accuracy'])
        diff_f1 = abs(results1['f1_score'] - results2['f1_score'])
        
        return {
            'accuracy_difference': diff_accuracy,
            'f1_difference': diff_f1,
            'significant': diff_f1 > 0.05  # Placeholder threshold
        }