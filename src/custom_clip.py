import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class CLIP_Contrastive(pl.LightningModule):
    """
    Custom CLIP wrapper that performs traning with captions
    """

    def __init__(
        self,
        clip_model,
        tokenized_captions,
        out_features = 1024,
        init_learning_rate = 1e-4,
    ):
        """
        Initialize the custom clip wrapper
        
        Parameters
        ----------
        model: one of the available clip models (RN50x16, ViT-B/32, etc.)
        image_encoder: the function that gets the image features
        text_encoder : the function that gets the text features
        learning_rate: the initial learning rate
        temperature  : a learnable param that scale the logits
        """
        super().__init__()
        self.model = clip_model.to(dtype=torch.float32)
        self.image_encoder = self.model.encode_image
        self.text_encoder  = self.model.encode_text
        self.learning_rate = init_learning_rate
        self.temperature   = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.caption_weights = self._caption_weights(tokenized_captions)
        self.caption_embeddings = self._caption_embeddings(tokenized_captions, 
                                                            out_features)

        # freeze the parameters of text_transformer to save memory
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        # freeze params of the token embeddings
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False

    def _caption_embeddings(self, tokenized_captions, out_feats):
        """ Compute embeddings for all classes and templates """

        n_classes, n_templates, _ = tokenized_captions.shape
        ce = torch.zeros((n_classes, n_templates, out_feats),
                         requires_grad = False,
                         device='cuda')

        for idx, captions in enumerate(tokenized_captions):
            text_features = self.text_encoder(captions)
            text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            ce[idx] = text_features.detach()
        return ce


    def _caption_weights(self, tokenized_captions):
        """ Initialize weights for each caption """

        num_classes, num_templates, _ = tokenized_captions.shape
        captions_weights = [1 / num_templates for _ in range(num_templates)]
        return nn.Parameter(torch.tensor(captions_weights,
                                         dtype = torch.float32,
                                         requires_grad = True,
                                         device='cuda'))[None, :, None]


    def forward(self, batch):
        """ Used for inference only """

        images, _ = batch
        return self.image_encoder(images)

    def training_step(self, batch, batch_idx):
        """ The complete training loop """
        
        # compute image and text features
        images, targets = batch
        image_features = self.image_encoder(images)
        text_features = self.caption_embeddings * self.caption_weights

        # normalize features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = torch.sum(text_features, axis=1)

        # compute loss
        image_logits = (image_features @ text_features.T) * torch.exp(self.temperature)
        loss = F.cross_entropy(image_logits, targets)
        self.log('train/loss', loss, on_epoch=True)
        
        # compute accuracy
        pred_labels = image_logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, targets)
        self.log('train/accuracy', acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """ The complete validation loop """

        # compute image and text features
        images, targets = batch
        image_features = self.image_encoder(images)
        text_features = self.caption_embeddings * self.caption_weights

        # normalize features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = torch.sum(text_features, axis=1)

        # compute loss
        image_logits = (image_features @ text_features.T) * torch.exp(self.temperature)
        loss = F.cross_entropy(image_logits, targets)
        self.log('val/loss', loss, on_epoch=True)

        # compute accuracy
        pred_labels = image_logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, targets)
        self.log('val/accuracy', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """ The complete test loop """

        # compute image and text features
        images, targets = batch
        image_features = self.image_encoder(images)
        text_features = self.caption_embeddings * self.caption_weights

        # normalize features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = torch.sum(text_features, axis=1)

        image_logits = (image_features @ text_features.T) * torch.exp(self.temperature)
        pred_labels = image_logits.argmax(dim=-1)
        return {'preds': pred_labels, 'labels': targets}

    def test_epoch_end(self, outputs):
        preds = torch.cat([o['preds'] for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        acc = torchmetrics.functional.accuracy(preds, labels)

        self.log('test/accuracy', acc)

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

        return {
            "optimizer": optimizer,
            "lr_scheduler": { 
                "scheduler": scheduler, 
                "monitor": "val/loss"},
            }

    def classify(self, image_batch):
        """ Performs classification """

        image_batch = image_batch.cuda()
        image_features = self.image_encoder(image_batch)
        text_features = torch.sum(self.caption_embeddings * self.caption_weights, axis=1)
        image_logits = torch.exp(self.T) * image_features @ text_features.T

        return image_logits.argmax(dim=-1)
    
    

class CLIP_Linear(pl.LightningModule):
    """
    Custom CLIP wrapper that uses the pretrained backbone of
    CLIP model and can be trained as an image classifier.
    Training is done as an ordinary classifier with CE loss.
    """
    def __init__(
        self,
        clip_model,
        num_classes=None,
        out_features=1024,
        freeze_visual=False,
        learning_rate=1e-4
    ):
        super().__init__()
        self.model = clip_model.to(dtype=torch.float32)
        self.image_encoder = self.model.encode_image
        self.learning_rate = learning_rate

        # freeze the image encoder, if needed
        if freeze_visual:
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # freeze the parameters of text_transformer to save memory
        for param in self.model.transformer.parameters():
            param.requires_grad = False

        # freeze params of the token embeddings
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False

        assert num_classes, "Number of classes has to be specified"
        self.classifier = nn.Linear(
            out_features, num_classes).to(dtype=torch.float32)

    def forward(self, batch):
        image_batch, _ = batch

        # Returns the embeddings of the images (either RN50 or ViT)
        image_features = self.image_encoder(image_batch)

        return self.classifier(image_features)

    def training_step(self, batch, batch_idx):
        """ The complete training loop """
        
        # Get the image features and classify
        image_batch, true_labels = batch
        image_features = self.image_encoder(image_batch)
        image_logits = self.classifier(image_features)
        
        loss = F.cross_entropy(image_logits, true_labels)
        self.log('train/loss', loss)

        pred_labels = image_logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
        self.log('train/accuracy', acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """ The complete validation loop """

        image_batch, true_labels = batch
        image_features = self.image_encoder(image_batch)
        image_logits = self.classifier(image_features)

        loss = F.cross_entropy(image_logits, true_labels)
        self.log('val/loss', loss, on_epoch=True)

        pred_labels = image_logits.argmax(dim=-1)
        acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
        self.log('val/accuracy', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """ The complete test loop """

        image_batch, true_labels = batch
        image_features = self.image_encoder(image_batch)
        image_logits = self.classifier(image_features)
        
        pred_labels = image_logits.argmax(dim=-1)
        return {'preds': pred_labels, 'labels': true_labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([o['preds'] for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        acc = torchmetrics.functional.accuracy(preds, labels)

        self.log('test/accuracy', acc)
    
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

        return {
            "optimizer": optimizer,
            "lr_scheduler": { 
                "scheduler": scheduler, 
                "monitor": "val/accuracy"},
            }

    def classify(self, image_batch):
        image_batch = image_batch.cuda()
        image_features = self.image_encoder(image_batch)
        return self.classifier(image_features).argmax(axis=-1)