import os

DATASET_DIR = '/oblivion/Datasets/FFHQ' # FFHQ dataset root path
CHECKPOINT_DIR = os.path.join(os.getcwd(), 'checkpoint') # pretrained model checkpoints

PRETRAINED_MODELS = {
    'stylegan1' : os.path.join(CHECKPOINT_DIR, 'res50_sg1.pth.tar'),
    'stylegan2' : os.path.join(CHECKPOINT_DIR, 'res50_sg2.pth.tar'),
    'sdv1.4'    : os.path.join(CHECKPOINT_DIR, 'res50_sdv1_4.pth.tar'),
    'sd1_4'    : os.path.join(CHECKPOINT_DIR, 'res50_sfhq_part2.pth.tar'),
}

IMAGE_DIR = {
    'real'      : os.path.join(DATASET_DIR, 'images1024x1024'),
    'stylegan1' : os.path.join(DATASET_DIR, 'generated', 'stylegan1-psi-0.5', 'images1024x1024'),
    'stylegan2' : os.path.join(DATASET_DIR, 'generated', 'stylegan2-psi-0.5', 'images1024x1024'),
    'stylegan_xl' : os.path.join(DATASET_DIR, 'generated', 'styleganxl-psi-0.5', 'images1024x1024'),
    'styleganxl' : os.path.join(DATASET_DIR, 'generated', 'styleganxl-psi-0.5', 'images1024x1024'),
    'sdv14'    : os.path.join(DATASET_DIR, 'generated', 'sdv1_4', 'images1024x1024'),
    'sdv1_4'    : os.path.join(DATASET_DIR, 'generated', 'sdv1_4', 'images1024x1024'),
}
