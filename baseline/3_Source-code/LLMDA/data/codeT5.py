from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# Function to load data from a JSON file
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    labels = [1 if item["name"] == "security" else 0 for item in data]  # 1 for "security", 0 otherwise
    return texts, labels

# Load training and test data

train_texts, train_labels = load_data('train.json')
test_texts, test_labels = load_data('test.json')

train_length = len(train_texts)//10
test_length = len(train_texts)//10
train_texts, train_labels = train_texts[:train_length], train_labels[:train_length]
test_texts, test_labels = test_texts[:test_length], test_labels[:test_length]
# import ipdb
# ipdb.set_trace()
# Function to generate embeddings
def generate_embeddings(data_subset):
    vectors = []
    max_length = 512  # Max sequence length

    with tqdm(total=len(data_subset)) as pbar:
        for text in data_subset:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            with torch.no_grad():
                outputs = model.encoder(**inputs)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            vectors.append(vector)
            pbar.update(1)

    return np.array(vectors)

# Generate embeddings for training and test data
train_vectors = generate_embeddings(train_texts)
test_vectors = generate_embeddings(test_texts)

# Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(train_vectors, train_labels)

# Predict probabilities on test data
test_probs = clf.predict_proba(test_vectors)[:, 1]

# Calculate AUC
auc = roc_auc_score(test_labels, test_probs)
fpr, tpr, thresholds = roc_curve(test_labels, test_probs)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Convert probabilities to binary predictions based on the optimal threshold
test_pred = (test_probs >= optimal_threshold).astype(int)

# Calculate other metrics
recall = recall_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)
accuracy = accuracy_score(test_labels, test_pred)
tn, fp, fn, tp = confusion_matrix(test_labels, test_pred).ravel()

# Positive and negative recalls
recall_p = tp / (tp + fn)
recall_n = tn / (tn + fp)

# Print metrics
print(f'ACC:{accuracy:.4f}, Precision:{precision:.4f}, AUC:{auc:.4f}, Recall:{recall:.4f}, Recall+:{recall_p:.4f}, Recall-:{recall_n:.4f}, F1:{f1:.4f}, TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}')
