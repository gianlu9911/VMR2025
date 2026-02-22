import os

DATASET_DIR = '/oblivion/Datasets/FFHQ' # FFHQ dataset root path
CHECKPOINT_DIR = os.path.join(os.getcwd(), '../checkpoint') # pretrained model checkpoints

PRETRAINED_MODELS = {
    'stylegan1' : os.path.join(CHECKPOINT_DIR, 'res50_sg1.pth.tar'),
    'stylegan2' : os.path.join(CHECKPOINT_DIR, 'res50_sg2.pth.tar'),
    'sdv1.4'    : os.path.join(CHECKPOINT_DIR, 'res50_sdv1_4.pth.tar'),
    'sd1_4'    : os.path.join(CHECKPOINT_DIR, 'res50_sfhq_part2.pth.tar'),
    'sdv1_4'    : os.path.join(CHECKPOINT_DIR, 'res50_sfhq_part2.pth.tar'),
    'stylegan_xl' : os.path.join(CHECKPOINT_DIR, 'stylegan_xl.pth'),
    'cyclegan_facades' : os.path.join('checkpoint', 'cyclegan_facades.pth'),
    'dogan_faces' : "checkpoint/resnet50_progan_finetuned.pth",
}
# SG1 -> SG2 -> SDV1.4 -> SG3 -> SGXL -> SD2.1
IMAGE_DIR = {
    'real'      : os.path.join(DATASET_DIR, 'images1024x1024'),
    'stylegan1' : os.path.join(DATASET_DIR, 'generated', 'stylegan1-psi-0.5', 'images1024x1024'),
    'stylegan2' : os.path.join(DATASET_DIR, 'generated', 'stylegan2-psi-0.5', 'images1024x1024'),
    'sdv1_4'    : os.path.join(DATASET_DIR, 'generated', 'sdv1_4', 'images1024x1024'),
    'stylegan3' : os.path.join(DATASET_DIR, 'generated', 'stylegan3-psi-0.5', 'images1024x1024'),
    'stylegan_xl' : os.path.join(DATASET_DIR, 'generated', 'styleganxl-psi-0.5', 'images1024x1024'),
    'sdv2_1'    : os.path.join(DATASET_DIR, 'generated', 'sdv2_1', 'images1024x1024'),  # Add path for sdv2_1
    #'real_DoGAN_facades' : '/seidenas/datasets/DoGANs/new/Pristine/CycleGAN/celeba256/',
    'real_DoGAN_faces_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/',
    'cyclegan_facades' : '/seidenas/datasets/DoGANs/new/Generated/CycleGAN/zebra2horse/',
    'progan_celeb256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/celeba256',
    'progan_1024_celebhq' : '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    'glow_smiling' : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
    'star_gan' : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Smiling',
}

IMAGE_DIR_DOGAN = {
    'real_progan256' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/',
    'fake_progan256' : '/seidenas/datasets/DoGANs/new/Generated/GGAN256/celeba256',
    'real_progan1024' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN1024/HQ-IMG/',
    'fake_progan1024' : '/seidenas/datasets/DoGANs/new/Generated/GGAN1024/celebhq',
    'real_stargan' : '/seidenas/datasets/DoGANs/new/Pristine/starGAN/celeba256',
    'fake_stargan' : '/seidenas/datasets/DoGANs/new/Generated/starGAN/Smiling',
    'real_glow' : '/seidenas/datasets/DoGANs/new/Pristine/GGAN256/celeba256/', #  glow non sembra avere real?????
    'fake_glow' : '/seidenas/datasets/DoGANs/new/Generated/glow/Male',
}