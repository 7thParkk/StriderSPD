import torch
import torch.nn as nn
import ipdb


class Transformer(nn.Module):

    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(True)

        # self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
        # self.dropout = nn.Dropout(0.5)
        # self.


    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        
        hiddens = raw_outputs.last_hidden_state
        # ipdb.set_trace()
        cls_feats = hiddens[:, 0, :] # 16, 768
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats)) # + 0.0001 *bias
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats) # 16,2
        outputs = {
            'predicts': predicts,        # 16, 2
            'cls_feats': cls_feats,      # 16, 768
            'label_feats': label_feats   # 16, 2, 768
        }
        return outputs
    


