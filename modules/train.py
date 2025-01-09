import torch
import json
import os
from dataset import prepare_ImageDataset
from unet import UNet
from trainer import UnetTrainer

# Load the configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Set the device
device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

# Load the dataset
train_dataset, val_dataset, train_dataloader, val_dataloader = prepare_ImageDataset(img_dir=config["dataset"]["4channels_img_dir"],
                                                                                    batch_size=config["dataset"]["batch_size"],
                                                                                    validation_split=config["dataset"]["validation_split"],
                                                                                    transform=None,
                                                                                    seed=123,
                                                                                    fraction=config["dataset"]["fraction"])

# Create the UNet model
model = UNet(input_channels=config["segmentation_model"]["input_channels"],
             output_channels=config["segmentation_model"]["output_channels"],
             base_channels=config["segmentation_model"]["base_channels"],
             channel_multiplier=config["segmentation_model"]["channel_multiplier"],
             num_resblocks=config["segmentation_model"]["num_resblocks"],
             num_groups=config["segmentation_model"]["num_groups"],
             downsampling_kernel_dim=config["segmentation_model"]["downsampling_kernel_dim"])

# Create the optimizer and the trainer
optimizer = torch.optim.Adam(model.parameters(), lr=config["training_configuration"]["learning_rate"])

trainer = UnetTrainer(model=model,
                      optimizer=optimizer,
                      device=device)

# Train the model
history = trainer.train(train_dataloader=train_dataloader,
                        num_epochs=config["training_configuration"]["num_epochs"],
                        save_folder=config["saving_configuration"]["save_folder"],
                        val_dataloader=val_dataloader,
                        grad_clip=config["training_configuration"]["grad_clip"])


# Save the model
model.save_model(save_folder=config["saving_configuration"]["save_folder"])
model.save_history(history=history, save_folder=config["saving_configuration"]["save_folder"])



