import os, gc
import json
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from data.get_dataset import GetDataset
from utils.helper import RunTime
from trainer.eval import evaluate
from config.logger import setup_logger
logger = setup_logger()
from utils.collect import custom_collate_fn
from utils.move_to_device import move_to_device
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator, DistributedDataParallelKwargs

def My_Train(accelerator, model, trainloader, optimizer, scheduler, criterion, args, cur_epoch):
    device = args.device
    if hasattr(accelerator, "gradient_accumulation_steps"):
        accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
    model.train()
    train_loss = 0.0
    bar = tqdm(trainloader, total=len(trainloader), desc="Epoch "+str(cur_epoch))
    step_counter = 0

    # 正常训练
    accum_counter = 0

    for i, data in enumerate(bar):
        data = move_to_device(data, args.device)
        with accelerator.autocast():
            logits = model(data)
            loss = criterion(logits.squeeze(dim=1).to(data['label'].device), data['label'].float())
            loss = loss.mean()
        accelerator.backward(loss)

        accum_counter += 1
        if accum_counter == accelerator.gradient_accumulation_steps or i == len(trainloader) - 1:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_counter = 0

        train_loss += accelerator.gather(loss).sum().item()


    train_loss /= len(trainloader.dataset)

    return model, train_loss

def train(args, model, node_tokenizer, seq_tokenizer):
    os.makedirs(args.save_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # 启用混合精度
        device_placement=False,  # Preserve the explicit GNN/HF model device map.
    )
    # load train data
    trainDataset = GetDataset(node_tokenizer, seq_tokenizer, args.root_path, args.ass_path, args.pse_path)
    train_data_size = len(trainDataset)
    train_sampler = RandomSampler(data_source=trainDataset)
    train_loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

    # load eval data
    evalDataset = GetDataset(node_tokenizer, seq_tokenizer, args.root_path, args.ass_path, args.pse_path, mode='dev')

    # optimizing choices
    args.max_steps = args.epoch * len(train_loader)
    print(f'Training loader length: {len(train_loader)}.')
    criterion = torch.nn.BCEWithLogitsLoss()
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=args.weight_decay)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1, num_training_steps=args.max_steps)

    best_acc, best_f1, best_recall, best_precision, best_fpr = 0.0, 0.0, 0.0, 0.0, 1.0
    logger.info("***** Running training *****")
    logger.info("Num examples = %d", train_data_size)
    logger.info("Num Epochs = %d", args.epoch)
    logger.info("Batch size = %d", args.batch_size)
    logger.info(RunTime())

    last_model = model

    # training!!!!
    print("Start training!!!") # here

    half_epoch = max(1, args.epoch // 2)   # 前/后阶段分界线
    
    for name, param in last_model.named_parameters():
        # 冻结 GNNE 内的 trans_encoder 和 token_embedding
        if name.startswith("gnn.trans_encoder.") or name.startswith("gnn.token_embedding.") or name.startswith("gnn.edge_embedding"):
            param.requires_grad_(False)
        # 冻结 LLM 中除了 q_proj_new 的部分（保留原有逻辑）
        elif name.startswith("llm.") and "q_proj_new" not in name:
            param.requires_grad_(False)
        # 其它参数都参与训练
        else:
            param.requires_grad_(True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, last_model.parameters()),
                    lr=args.learning_rate,
                    eps=args.adam_epsilon,
                    weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_training_steps = steps_per_epoch * args.epoch
    num_warmup_steps = int(total_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )
    last_model, optimizer, train_loader, scheduler = accelerator.prepare(
        last_model, optimizer, train_loader, scheduler
    ) 

    for cur_epoch in range(args.epoch):
        torch.cuda.empty_cache()
        gc.collect()
        epoch_train_start = time.perf_counter()
        last_model, loss = My_Train(accelerator, last_model, train_loader, optimizer, scheduler, criterion, args, cur_epoch)
        epoch_train_seconds = time.perf_counter() - epoch_train_start
        processed_examples = len(train_loader) * args.batch_size
        seconds_per_example = epoch_train_seconds / max(processed_examples, 1)
        timing_message = (
            f'TRAIN_EPOCH_TIMING epoch={cur_epoch} seconds={epoch_train_seconds:.3f} '
            f'batches={len(train_loader)} examples={processed_examples} '
            f'seconds_per_example={seconds_per_example:.6f}'
        )
        print(timing_message)
        logger.info(timing_message)
        logger.info(f'epoch {cur_epoch}, train loss = {round(loss, 3)}')

        result = evaluate(model=last_model, eval_dataset=evalDataset, args=args, accelerator=accelerator)

        epoch_checkpoint_name = (
            f'model_epoch{cur_epoch}_dev_acc_{result["accuracy"]:.6f}.pth'
        )
        epoch_record = {
            'epoch': cur_epoch,
            'train_loss': loss,
            'train_seconds': epoch_train_seconds,
            'train_examples': processed_examples,
            'seconds_per_example': seconds_per_example,
            **result,
            'checkpoint': epoch_checkpoint_name if args.save_every_epoch else None,
        }

        if accelerator.is_main_process:
            with open(os.path.join(args.save_path, 'epoch_metrics.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_record, ensure_ascii=False) + '\n')
            if args.save_every_epoch:
                torch.save(
                    accelerator.unwrap_model(last_model).state_dict(),
                    os.path.join(args.save_path, epoch_checkpoint_name),
                )

        if best_acc < result['accuracy']:
            best_acc = result['accuracy']
            logger.info(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
            print(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
            acc = result['accuracy']
            print(f'best_acc={round(acc,4)}')

            if accelerator.is_main_process:
                if args.save_every_epoch:
                    with open(os.path.join(args.save_path, 'best_checkpoint.txt'), 'w', encoding='utf-8') as f:
                        f.write(epoch_checkpoint_name + '\n')
                else:
                    for file in os.listdir(args.save_path):
                        if file.startswith("model_epoch") and file.endswith("_best_acc.pth"):
                            os.remove(os.path.join(args.save_path, file))
                    torch.save(
                        accelerator.unwrap_model(last_model).state_dict(),
                        os.path.join(args.save_path, f'model_epoch{str(cur_epoch)}_best_acc.pth'),
                    )
        logger.info(RunTime())

    # finishing training!!!
    logger.info('*********************************************************************************************')
    logger.info(f'The best accuary in the training steps is {round(best_acc,4)}')
    logger.info('*********************************************************************************************')

    logger.info(f'Save the last model!')
    # torch.save(last_model.state_dict(), os.path.join(args.save_path, f'model_epoch{str(args.epoch-1)}_last.pth'))
    logger.info(f'Finishing training!')

# import os, gc
# import logging
# import torch
# import torch.nn as nn
# from torch.utils.data import RandomSampler
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from torch.optim import AdamW
# from tqdm import tqdm
# from data.get_dataset import GetDataset
# from utils.helper import RunTime
# from trainer.eval import evaluate
# from config.logger import setup_logger
# logger = setup_logger()
# from utils.collect import custom_collate_fn
# from utils.move_to_device import move_to_device
# from bitsandbytes.optim import AdamW as AdamW_8bit
# from torch.cuda.amp import autocast, GradScaler
# from accelerate import Accelerator, DistributedDataParallelKwargs

# def My_Train(accelerator, model, trainloader, optimizer, scheduler, criterion, args, cur_epoch):
#     device = args.device
#     if hasattr(accelerator, "gradient_accumulation_steps"):
#         accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
#     model.train()
#     train_loss = 0.0
#     bar = tqdm(trainloader, total=len(trainloader), desc="Epoch "+str(cur_epoch))
#     step_counter = 0

#     # 正常训练
#     accum_counter = 0

#     for i, data in enumerate(bar):
#         with accelerator.autocast():
#             logits = model(data)
#             loss = criterion(logits.squeeze(dim=1).to(data['label'].device), data['label'].float())
#             loss = loss.mean()
#         accelerator.backward(loss)

#         accum_counter += 1
#         if accum_counter == accelerator.gradient_accumulation_steps or i == len(trainloader) - 1:
#             accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad()
#             accum_counter = 0

#         train_loss += accelerator.gather(loss).sum().item()


#     train_loss /= len(trainloader.dataset)

#     return model, train_loss

# def train(args, model, node_tokenizer, seq_tokenizer):
#     if not os.path.exists(args.save_path):
#         os.mkdir(args.save_path)

#     accelerator = Accelerator(
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         mixed_precision="bf16",  # 启用混合精度
#     )
#     # load train data
#     trainDataset = GetDataset(node_tokenizer, seq_tokenizer, args.root_path, args.ass_path, args.pse_path)
#     train_data_size = len(trainDataset)
#     train_sampler = RandomSampler(data_source=trainDataset)
#     train_loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)

#     # load eval data
#     evalDataset = GetDataset(node_tokenizer, seq_tokenizer, args.root_path, args.ass_path, args.pse_path, mode='dev')

#     # optimizing choices
#     args.max_steps = args.epoch * len(train_loader)
#     print(f'Training loader length: {len(train_loader)}.')
#     criterion = torch.nn.BCEWithLogitsLoss()
#     # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=args.weight_decay)
#     # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1, num_training_steps=args.max_steps)

#     best_acc, best_f1, best_recall, best_precision, best_fpr = 0.0, 0.0, 0.0, 0.0, 1.0
#     logger.info("***** Running training *****")
#     logger.info("Num examples = %d", train_data_size)
#     logger.info("Num Epochs = %d", args.epoch)
#     logger.info("Batch size = %d", args.batch_size)
#     logger.info(RunTime())

#     last_model = model

#     # training!!!!
#     print("Start training!!!") # here

#     half_epoch = max(1, args.epoch // 2)   # 前/后阶段分界线
    
#     for name, param in last_model.named_parameters():
#         # 冻结 GNNE 内的 trans_encoder 和 token_embedding
#         if name.startswith("gnn.trans_encoder.") or name.startswith("gnn.token_embedding.") or name.startswith("gnn.edge_embedding"):
#             param.requires_grad_(False)
#         # 冻结 LLM 中除了 q_proj_new 的部分（保留原有逻辑）
#         elif name.startswith("llm.") and "q_proj_new" not in name:
#             param.requires_grad_(False)
#         # 其它参数都参与训练
#         else:
#             param.requires_grad_(True)

#     optimizer = AdamW(filter(lambda p: p.requires_grad, last_model.parameters()),
#                     lr=args.learning_rate,
#                     eps=1e-6,
#                     weight_decay=args.weight_decay)
#     steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
#     total_training_steps = steps_per_epoch * args.epoch
#     num_warmup_steps = int(total_training_steps * 0.1)

#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=total_training_steps
#     )
#     last_model, optimizer, train_loader, scheduler = accelerator.prepare(
#         last_model, optimizer, train_loader, scheduler
#     ) 

#     for cur_epoch in range(args.epoch):
#         torch.cuda.empty_cache()
#         gc.collect()
#         last_model, loss = My_Train(accelerator, last_model, train_loader, optimizer, scheduler, criterion, args, cur_epoch)
#         logger.info(f'epoch {cur_epoch}, train loss = {round(loss, 3)}')

#         result = evaluate(model=last_model, eval_dataset=evalDataset, args=args)

#         if best_acc < result['accuracy']:
#             best_acc = result['accuracy']
#             logger.info(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
#             print(f'best_accuracy change into {best_acc} at epoch {cur_epoch}')
#             acc = result['accuracy']
#             print(f'best_acc={round(acc,4)}')

#             for file in os.listdir(args.save_path):
#                 if file.startswith("model_epoch") and file.endswith("_best_acc.pth"):
#                     os.remove(os.path.join(args.save_path, file))

#             torch.save(last_model.state_dict(), os.path.join(args.save_path, f'model_epoch{str(cur_epoch)}_best_acc.pth'))
#         logger.info(RunTime())

#     # finishing training!!!
#     logger.info('*********************************************************************************************')
#     logger.info(f'The best accuary in the training steps is {round(best_acc,4)}')
#     logger.info('*********************************************************************************************')

#     logger.info(f'Save the last model!')
#     # torch.save(last_model.state_dict(), os.path.join(args.save_path, f'model_epoch{str(args.epoch-1)}_last.pth'))
#     logger.info(f'Finishing training!')
