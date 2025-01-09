import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import dice_coefficient, checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR


class UnetTrainer():

    def __init__(self,
                 model,
                 optimizer,
                 device):
        """Unet trainer for segmentation tasks.

        Parameters:
        -----------
        model : nn.Module
            UNet model.
        optimizer : torch optimizer
            Optimizer.
        device : torch device
            Device to use.
        """
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device


    def train(self,
              train_dataloader,
              num_epochs,
              save_folder,
              val_dataloader=None,
              grad_clip=None,
              cosine_annealing_scheduler=False):
        """Train the UNet model.

        Parameters:
        -----------
        train_dataloader : torch DataLoader
            Training data loader.
        num_epochs : int
            Number of epochs.
        save_folder : str
            Folder to save the model checkpoints. 
        val_dataloader : torch DataLoader
            Validation data loader. Default is None.
        grad_clip : float
            Gradient clipping. Default is None.
        cosine_annealing_scheduler : bool
            Use cosine annealing learning rate scheduler. Default is False.

        Returns:
        --------
        history : dict
            Training and validation history.
        """

        history = {'train_loss': [],
                   'val_loss': [],
                   'train_dice_coefficient': [],
                   'val_dice_coefficient': []}
        
        if cosine_annealing_scheduler:
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        for epoch in range(num_epochs):

            # Training mode
            self.model.train()

            # Train one epoch
            train_loss, train_dice = self._train_one_epoch(train_dataloader=train_dataloader,
                                                           epoch=epoch,
                                                           grad_clip=grad_clip)

            # Update history
            history['train_loss'].append(train_loss)
            history['train_dice_coefficient'].append(train_dice)

            if val_dataloader is not None:
                # Validation mode
                self.model.eval()

                # Evaluate the model
                val_loss, val_dice = self._validate(val_dataloader)

                # Update history
                history['val_loss'].append(val_loss)
                history['val_dice_coefficient'].append(val_dice)

            # Checkpoint
            checkpoint(model=self.model,
                       save_folder=save_folder,
                       current_epoch=epoch,
                       epoch_step=5)
            
            if cosine_annealing_scheduler:
                scheduler.step()

        return history
    

    def _train_one_epoch(self,
                         train_dataloader,
                         epoch,
                         grad_clip):
        
        running_train_loss = 0.0
        running_dice = 0.0

        mean_train_loss = 0.0
        mean_dice = 0.0

        self.optimizer.zero_grad()

        with tqdm(train_dataloader, unit='batches') as tepoch:

            for batch_idx, batch in enumerate(tepoch):

                # Update the progress bar description
                tepoch.set_description(f'Epoch {epoch+1}')

                # Load images to device
                imgs, masks = batch
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                pred = self.model(imgs)

                # Compute the loss
                loss = F.binary_cross_entropy(pred, masks, reduction='mean')
                with torch.no_grad():
                    binary_pred = (pred > 0.5).float()
                    dice = dice_coefficient(binary_pred, masks)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # Update the model parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update the running losses and mean losses
                running_train_loss += loss.item()
                running_dice += dice.item()

                mean_train_loss = running_train_loss / (batch_idx + 1)
                mean_dice = running_dice / (batch_idx + 1)

                # Update the progress bar
                tepoch.set_postfix(total_loss="{:.6f}".format(mean_train_loss),
                                   dice="{:.3f}".format(mean_dice))
        
        return mean_train_loss, mean_dice
    

    def _validate(self, 
                  val_dataloader):
        
        running_val_loss = 0.0
        running_dice = 0.0

        mean_val_loss = 0.0
        mean_dice = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):

                # Load images to device
                imgs, masks = batch
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                pred = self.model(imgs)

                # Compute the loss
                loss = F.binary_cross_entropy(pred, masks, reduction='mean')
                binary_pred = (pred > 0.5).float()
                dice = dice_coefficient(binary_pred, masks)

                # Update the running losses and mean losses
                running_val_loss += loss.item()
                running_dice += dice.item()
            
        mean_val_loss = running_val_loss / len(val_dataloader)
        mean_dice = running_dice / len(val_dataloader)

        print(f'Validation loss: {mean_val_loss:.6f}')
        print(f'Validation dice coefficient: {mean_dice:.3f}')
        
        return mean_val_loss, mean_dice
    



