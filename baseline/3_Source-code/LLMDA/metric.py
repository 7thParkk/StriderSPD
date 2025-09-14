from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve,confusion_matrix
import torch


def result():
    y_pred = torch.load('best.y.pred')
    y_true = torch.load('best.y.true')

    print(f'length of y_pred{len(y_pred)}')

    # y_true = [i.tolist() for i in y_true]
    # y_pred = [i.tolist() for i in y_pred]

    # y_true = [item for sublist in y_true for item in sublist]
    # y_pred = [item for sublist in y_pred for item in sublist]

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print(f'ACC:{accuracy:.4f}, Precision:{precision:.4f} AUC:{auc:.4f}, Recall:{recall:.4f} Recall+:{recall_p:.4f}, Recall-:{recall_n:.4f}, F1:{f1:.4f}, TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}')



if __name__=='__main__':
    y_pred = torch.load('best.y.pred')
    y_true = torch.load('best.y.true')

    print(f'length of y_pred{len(y_pred)}')

    y_true = [i.tolist() for i in y_true]
    y_pred = [i.tolist() for i in y_pred]

    y_true = [item for sublist in y_true for item in sublist]
    y_pred = [item for sublist in y_pred for item in sublist]

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    
    tn, fp, fn, tp = confusion_matrix(y_pred,y_true).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print(f'ACC:{accuracy:.4f}, Precision:{precision:.4f} AUC:{auc:.4f}, Recall:{recall:.4f} Recall+:{recall_p:.4f}, Recall-:{recall_n:.4f}, F1:{f1:.4f}, TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}')
