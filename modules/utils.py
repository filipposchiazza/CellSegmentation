import os
import cv2

def checkpoint(model, save_folder, current_epoch, epoch_step):
    """Save checkpoints of the model every epoch_step.

    Parameters:
    -----------
    model : torch.nn.Module
        Model to save.
    save_folder : str
        Folder to save the model.
    current_epoch : int
        Current epoch.
    epoch_step : int
        Epoch step to save the model.
    """

    if current_epoch != 0 and current_epoch % epoch_step == 0:
        # Create the checkpoint directory
        checkpoint_dir = os.path.join(save_folder, 'checkpoints', f'epoch_{current_epoch}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model
        model.save_model(checkpoint_dir)
    return True



def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Calculate the Dice Coefficient between the predicted and target masks.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask.
    target : torch.Tensor
        Target mask.
    epsilon : float
        Small value to avoid division by zero.
    
    Returns
    -------
    dice_coeff : float
        Dice Coefficient.
    """
    # Flatten the tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    numerator = 2. * intersection + epsilon
    denominator = pred_flat.sum() + target_flat.sum() + epsilon
    dice_coeff = numerator / denominator
    return dice_coeff



def draw_countour(image, mask, color=(0.0, 1.0, 0.0)):
    """Draw the contour of the mask on the image.

    Parameters:
    -----------
    image : np.ndarray
        Image.
    mask : np.ndarray
        Mask.
    color : tuple
        Color of the contour.
    
    Returns:
    --------
    image : np.ndarray
        Image with the contour.
    """
    mask = mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_img = image.copy()
    result_img = cv2.drawContours(result_img, contours, -1, color, 2)
    return result_img 