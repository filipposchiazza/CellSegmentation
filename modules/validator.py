import torch
from utils import dice_coefficient
from tqdm import tqdm


class UNetValidator():

    def __init__(self, model, device):
        """Unet validator for segmentation tasks.

        Parameters:
        -----------
        model : nn.Module
            UNet model.
        device : torch device
            Device to use.
        """
        
        self.model = model.to(device)
        self.device = device


    
    def validate(self, dataset, num_samples):
        """Validate the UNet model.

        Parameters:
        -----------
        dataset : torch Dataset
            Dataset to evaluate.
        num_samples : int
            Number of samples to use for validation.

        Returns:
        --------
        dice_list : list
            List of dice coefficients
        """
        
        self.model.eval()
        dice_list = []
        num_iterations = min(num_samples, len(dataset))
        
        with torch.no_grad():
            for i in tqdm(range(num_iterations)):
                images, masks = dataset[i]
                images = images.unsqueeze(0).to(self.device)
                masks = masks.unsqueeze(0).to(self.device)
                
                outputs = self.model(images)
                dice = dice_coefficient(outputs, masks)
                dice_list.append(dice.item())

        return dice_list
    

    def validate_inter_models(self, model_2, dataset, num_samples):
        """Validate the UNet model.

        Parameters:
        -----------
        model_2 : nn.Module
            Second UNet model.
        dataset : torch Dataset
            Dataset to evaluate.
        num_samples : int
            Number of samples to use for validation.

        Returns:
        --------
        dice_list : list
            List of dice coefficients
        """
        self.model.eval()
        model_2.eval()
        dice_list = []
        num_iterations = min(num_samples, len(dataset))

        with torch.no_grad():
            for i in tqdm(range(num_iterations)):
                images, masks = dataset[i]
                images = images.unsqueeze(0).to(self.device)
                masks = masks.unsqueeze(0).to(self.device)

                outputs_1 = self.model(images)
                outputs_2 = model_2(images)

                dice = dice_coefficient(outputs_1, outputs_2)
                dice_list.append(dice.item())

        return dice_list
    