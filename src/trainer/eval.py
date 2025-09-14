import os
import logging
import torch
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, recall_score
from config.logger import setup_logger
logger = setup_logger()
from utils.move_to_device import move_to_device
from utils.collect import custom_collate_fn
from accelerate import Accelerator


# def evaluate(model, eval_dataset, args):
#     eval_sampler = SequentialSampler(data_source=eval_dataset)
#     evalloader = DataLoader(dataset=eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=custom_collate_fn, sampler=eval_sampler, drop_last=True)

#     device = args.device
#     model.eval()

#     preds = []
#     labels = []

#     logger.info("***** Running evaluation *****")
#     eval_acc, eval_f1, eval_precision, eval_fpr = 0.0, 0.0, 0.0, 1.0

#     for data in evalloader:
#         data = move_to_device(data, device)
#         with torch.no_grad():
#             logits = model.forward(data)
#         print("logits")
#         print("logits shape:", logits.shape)  # e.g. [batch_size, 1]
#         print(logits[:, 0])
#         probs = torch.sigmoid(logits[:, 0])
#         pred = logits[:, 0] >=0.5
#         preds.extend(pred.int().tolist())
#         # pred = torch.argmax(logits, dim=1)
#         # preds.extend(pred.tolist())
#         labels.extend(data['label'].int().tolist())
#     logger.info("label 结果")
#     logging.info(labels)
#     logger.info("preds 结果")
#     logging.info(preds)

#     eval_acc = accuracy_score(labels,preds)
#     eval_precision = precision_score(labels,preds)
#     eval_recall = recall_score(labels, preds)
#     eval_f1 = f1_score(labels, preds)

#     conf_matrix = confusion_matrix(labels, preds, labels=[0,1])
#     TN = conf_matrix[0, 0]
#     FP = conf_matrix[0, 1]
#     eval_fpr = FP / ((FP + TN)+0.0001)

#     result = {
#         'accuracy': eval_acc,
#         'precision': eval_precision,
#         'recall': eval_recall,
#         'f1': eval_f1,
#         'fpr': eval_fpr
#     }
    
#     for key in sorted(result.keys()):
#         logger.info("%s = %s", key, str(result[key]))
#     logger.info("" + "*" * 30)
#     print(f'eval_acc={round(eval_acc,4)}, eval_f1={round(eval_f1, 4)}, eval_precision={round(eval_precision,4)}, eval_fpr={round(eval_fpr, 4)}')

#     return result

def evaluate(model, eval_dataset, args):
    # 初始化accelerator (与训练配置一致)
    accelerator = Accelerator(
        mixed_precision="bf16"
    )
    
    eval_sampler = SequentialSampler(data_source=eval_dataset)
    evalloader = DataLoader(
        dataset=eval_dataset, 
        batch_size=args.eval_batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        sampler=eval_sampler
    )
    
    # 准备模型和数据加载器
    model, evalloader = accelerator.prepare(model, evalloader)
    
    model.eval()
    preds = []
    labels = []

    logger.info("***** Running evaluation *****")
    
    for data in evalloader:
        with torch.no_grad():
            # 自动处理混合精度
            with accelerator.autocast():
                logits = model(data)
            probs = torch.sigmoid(logits[:, 0])
            pred = probs >= 0.5
            preds.extend(pred.int().cpu().tolist())
            labels.extend(data['label'].int().cpu().tolist())
    logger.info("label 结果")
    logging.info(labels)
    logger.info("preds 结果")
    logging.info(preds)
    # 指标计算（保持原有逻辑）
    eval_acc = accuracy_score(labels, preds)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    
    conf_matrix = confusion_matrix(labels, preds, labels=[0,1])
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    eval_fpr = FP / ((FP + TN) + 0.0001)
    
    result = {
        'accuracy': eval_acc,
        'precision': eval_precision,
        'recall': eval_recall,
        'f1': eval_f1,
        'fpr': eval_fpr
    }
    
    for key in sorted(result.keys()):
        logger.info("%s = %s", key, str(result[key]))
    logger.info("" + "*" * 30)
    print(f'eval_acc={round(eval_acc,4)}, eval_f1={round(eval_f1, 4)}, '
          f'eval_precision={round(eval_precision,4)}, eval_fpr={round(eval_fpr, 4)}')

    return result