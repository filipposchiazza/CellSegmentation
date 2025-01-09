import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset configuration

IMG_DIRS = ['/mnt/local/work/filippo.schiazza/citology_patches_with_masks/CTR',
            '/mnt/local/work/filippo.schiazza/citology_patches_with_masks/MDS',
            '/mnt/local/work/filippo.schiazza/citology_patches_with_masks/AML']

SYN_IMG_DIRS = ['/mnt/local/work/filippo.schiazza2/gen_imgs/round_1/Syn/CTR',
                '/mnt/local/work/filippo.schiazza2/gen_imgs/round_1/Syn/MDS',
                '/mnt/local/work/filippo.schiazza2/gen_imgs/round_1/Syn/AML']
BATCH_SIZE = 8
FRACTION = 0.2
VALIDATION_SPLIT = 0.05
TEST_SPLIT = 0.20
SYN_FRACTION = 1.0
SYN_VALIDATION_SPLIT = 0.05
SYN_TEST_SPLIT = 0.0


# model configuration
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
BASE_CHANNELS = 32
CHANNEL_MULTIPLIER = [1, 2, 4, 8, 16]
NUM_RES_BLOCKS = [1, 2, 2, 3, 4]
NUM_GROUPS = 8
DOWNSAMPLING_KERNEL_DIM = 2

# training configuration
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
GRAD_CLIP = 1.0

# saving configuration
SAVE_FOLDER_REAL = '/mnt/local/work/filippo.schiazza2/seg_models/01'
SAVE_FOLDER_SYN = '/mnt/local/work/filippo.schiazza2/seg_models/02'
EVALUATION_RESULTS_FOLDER = '/mnt/local/work/filippo.schiazza2/Imgs4Paper/Features'
