import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tabulate
import torch

import argparse
from torch import nn

from config import PRETRAINED_MODELS, IMAGE_DIR
from src.dataloader import RealSynthethicDataloader
from src.net import load_pretrained_model

from torch.utils.data import DataLoader
from tqdm import tqdm

def accuracy(output, labels):
    with torch.no_grad():
        batch_size = labels.size(0)

        _, predicted = torch.max(output.data, 1)
        res = (predicted == labels).sum().item() / batch_size
    return res

def evaluate(dataloader, model, criterion, args, print_freq=10):
    # switch to evaluate mode
    model.eval()

    val_loss = 0
    val_acc = 0
    k = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit=' batch', desc='Evaluation on test_set: ') as bar:
            for i, (images, target) in enumerate(dataloader):
                bar.update(1)
                if args.device is not None:
                    images = images.to(args.device)
                target = target.to(args.device)

                # compute output
                output = model(images)
                val_loss_it = criterion(output, target)
                val_acc_it = accuracy(output, target)

                val_acc += val_acc_it * target.size(0)
                val_loss += val_loss_it.item() * target.size(0)

                k += target.size(0)  # no of samples already processed

                if i % print_freq == 0:
                    bar.set_postfix({'batch_test_loss' : round(val_loss_it.item(), 5),
                                     'batch_test_acc' : round(val_acc_it, 5),
                                     'test_loss' : round(val_loss / k, 5),
                                     'test_acc' : round(val_acc / k, 5)
                                     })

            bar.close()

    val_loss /= len(dataloader.sampler)
    val_acc /= len(dataloader.sampler)

    return val_loss, val_acc

########### MAIN ###########
def main(args):
    if args.device is not None and args.device != '':
        if torch.cuda.is_available():
            print('Use GPU: {}.\n'.format(args.device))
            args.device = 'cuda:{}'.format(args.device)
        else:
            print('GPUs not available. Using CPU')
            args.device = 'cpu'
            args.num_workers = 1
    else:
        print('Using CPU')
        args.device = 'cpu'
        args.num_workers = 1
    #

    ck_fn = PRETRAINED_MODELS[args.train]
    model = load_pretrained_model(ck_fn) #, args.device 
    model.to(args.device)
    print(model)

    if args.test not in list(IMAGE_DIR.keys()):
        raise Exception('{} not found in {}'.format(args.test, list(IMAGE_DIR.keys())))

    real_dir = IMAGE_DIR['real']
    fake_dir = IMAGE_DIR[args.test]

    if args.train == args.test:
        print('[IN-DATASET] Evaluation from {} to {}'.format(args.train, args.test))
    else:
        print('[CROSS-DATASET] Evaluation from {} to {}'.format(args.train, args.test))

    #
    print('\nReal RGB: ' + real_dir)
    print('Fake RGB: ' + fake_dir + '\n')
    #

    rgb_real = IMAGE_DIR['real']
    rgb_fake = IMAGE_DIR[args.test]

    test_ds = RealSynthethicDataloader(rgb_real, rgb_fake)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss().to(args.device)
    _, acc = evaluate(test_loader, model, criterion, args)

    print('Done\n')

    headers = ['TRAIN', 'TEST', 'ACC']
    table_str = tabulate.tabulate([[args.train, args.test, acc]],
                                  headers=headers,
                                  tablefmt='outline', floatfmt='.6f',
                                  colalign=tuple(['center'] * len(headers)))
    print(table_str + '\n\nDone.\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Command line parameters
    parser.add_argument('--train', type=str, default='stylegan1', help='Retrieve pretrained model on which is trained on')
    parser.add_argument('--test', type=str, default='stylegan2', help='Test set')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='0')

    main(parser.parse_args())