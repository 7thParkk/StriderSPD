import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
import ipdb
from metric import result
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve,confusion_matrix
import time  
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)


        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        start_time = time.time()  # 记录开始时间
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        y_true = []
        y_pred = []
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()} 
            targets = targets.to(self.args.device) 
            
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0) 
            y_true.append(targets)
            y_pred.append(torch.argmax(outputs['predicts'], -1))

            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            
            n_train += targets.size(0)
        train_time = time.time() - start_time  # 计算训练时间
        return train_loss / n_train, n_correct / n_train, train_time  # 返回训练时间

    def _test(self, dataloader, criterion):
        start_time = time.time()  # 记录开始时间
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)
                y_true.append(targets)
                y_pred.append(torch.argmax(outputs['predicts'], -1))
        
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
        #TN,FP,FN,TP
        tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        
        # 将评估指标输出到log
        self.logger.info(f'ACC:{accuracy:.4f}, Precision:{precision:.4f} AUC:{auc:.4f}, Recall:{recall:.4f}')
        self.logger.info(f'Recall+:{recall_p:.4f}, Recall-:{recall_n:.4f}, F1:{f1:.4f}')
        self.logger.info(f'Confusion Matrix: TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp}')
        
        test_time = time.time() - start_time  # 计算测试时间
        return test_loss / n_test, n_correct / n_test, recall, y_true, y_pred, self.model.state_dict(), test_time  # 返回测试时间

    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
        best_loss, best_acc, best_recall = 0, 0, 0
        best_y_true, best_y_pred = [], []
        bestmd = self.model.state_dict()
        for epoch in range(self.args.num_epoch):
            epoch_start_time = time.time()  # 记录epoch开始时间
            train_loss, train_acc, train_time = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_recall, y_true, y_pred, modelstatedict, test_time = self._test(test_dataloader, criterion)
            epoch_time = time.time() - epoch_start_time  # 计算epoch总时间
            
            if test_recall > best_recall: 
                best_acc, best_loss, best_recall = test_acc, test_loss, test_recall
                best_y_true, best_y_pred = y_true, y_pred
                bestmd = modelstatedict
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}, time: {:.2f}s'.format(train_loss, train_acc*100, train_time))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, time: {:.2f}s'.format(test_loss, test_acc*100, test_time))
            self.logger.info('[test] loss: {:.4f}, recall: {:.2f}'.format(test_loss, test_recall*100))
            self.logger.info('[epoch] total time: {:.2f}s'.format(epoch_time))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}, best recall: {:.2f}'.format(best_loss, best_acc*100, best_recall*100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        torch.save(best_y_true, 'best.y.true')
        torch.save(best_y_pred, 'best.y.pred')
        torch.save(bestmd,'bestmodel.h5')


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
    result()

