import torch 
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

from .transformer import TransformerEncoder

class TransformerPredictor(L.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout, input_dropout):
        super().__init__()
        self.save_hyperparameters()

        self.input_net = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Linear(input_dim, model_dim)
        )

        # self.poistional_encoding = PositionalEncoding(

        # )
        
        self.transformer = TransformerEncoder(
            num_layers = num_layers,
            input_dim = model_dim,
            dim_forward = 2*model_dim,
            num_heads = num_heads,
            dropout = dropout
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=False):
        x = self.input_net(x)
        # if add_positional_encoding:
        #     x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x
    
    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        x = self.input_net(x)
        # if add_positon_encoding:
        #     x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=None)
        return attention_maps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError



class AnomalyPredictor(TransformerPredictor):
    def _calculator_loss(self, batch, mode="train"):
        img_sets, labels = batch
        preds = self.forward(img_sets)
        preds = preds.squeeze(-1)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculator_loss(batch, mode="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
         _ = self._calculator_loss(batch, mode="val")
        
    def test_step(self, batch, batch_idx):
        _ = self._calculator_loss(batch, mode="test")
        


