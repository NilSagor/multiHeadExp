import torch 
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy








class DummyAnomalyModel(L.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes       
        self.dummy_param = nn.Parameter(torch.randn(num_classes))
        self.loss_module = nn.CrossEntropyLoss()

        # self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        # self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        # self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        
        # # Register metrics as model submodules so Lightning moves them to device
        # self.add_module('train_acc', self.train_acc)
        # self.add_module('val_acc', self.val_acc)
        # self.add_module('test_acc', self.test_acc)

    # def setup(self, stage=None):
    #     # Metrics are created AFTER model is moved to device
    #     self.train_acc = MulticlassAccuracy(num_classes=self.num_classes)
    #     self.val_acc = MulticlassAccuracy(num_classes=self.num_classes)
    #     self.test_acc = MulticlassAccuracy(num_classes=self.num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        return self.dummy_param.expand(batch_size, -1)
        # return torch.randn(batch_size, self.num_classes, device=x.device, dtype=x.dtype)
    
    def _calculate_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds==labels).float().mean()

    def training_step(self, batch, batch_idx):
        img_sets, labels = batch
        # print(f"Model device: {next(self.parameters()).device}")
        # print(f"Input device: {img_sets.device}") 
        logits = self(img_sets)
        loss = self.loss_module(logits, labels)
        # acc = self.train_acc(logits, labels)
        acc = self._calculate_accuracy(logits, labels)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img_sets, labels = batch 
        logits = self(img_sets)        
        acc = self._calculate_accuracy(logits, labels)
        # acc = self.val_acc(logits, labels)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        img_sets, labels = batch 
        logits = self(img_sets)        
        # acc = self.test_acc(logits, labels)
        acc = self._calculate_accuracy(logits, labels)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD([self.dummy_param], lr=0.01)
        

