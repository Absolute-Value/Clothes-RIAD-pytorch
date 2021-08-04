#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, datetime, argparse, csv
from modules.unet import UNet
from modules.initializer import init_weight
from modules.loader import get_loader
from modules.logger import get_logger
from modules.funcs import EarlyStop, gen_mask
from losses.gms_loss import MSGMS_Loss
from losses.ssim_loss import SSIM_Loss

parser = argparse.ArgumentParser(description='RIAD anomaly detection')
parser.add_argument('--dataset', type=str, default='mvtec')
parser.add_argument('--data_type', type=str, default='toothbrush')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300, help='maximum training epochs')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--val_rate', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of Adam')
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--seed', type=int, default=999, help='manual seed')
parser.add_argument('--lambda_G', type=float, default=1)
parser.add_argument('--lambda_S', type=float, default=1)
parser.add_argument('--k_value', type=int, nargs='+', default=[2, 4, 8, 16])
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

dt_now = datetime.datetime.now()
result_dir = 'result/{}/'.format(args.dataset) + '{}{}_seed{}_'.format(args.data_type, args.img_size, args.seed) + dt_now.strftime('%Y%m%d_%H%M%S')
pic_dir = result_dir + '/pic'
if not os.path.isdir(pic_dir):
    os.makedirs(pic_dir)

logger = get_logger(result_dir,'train.log')
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
logger.info("devece : {}".format(device))

state = {k: v for k, v in args._get_kwargs()}
logger.info(state)

train_loader, val_loader = get_loader(config=args.dataset, class_name=args.data_type, img_size=args.img_size, is_train=True, batch_size=args.batch_size, val_rate=args.val_rate)

net = UNet().to(device)
init_weight(net, init_type=args.init_type)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
logger.info(net)

## Train
vis_loader = val_loader
vis_inputs_o = torch.cat([batch[0] for batch in vis_loader], dim=0)
vis_inputs = vis_inputs_o[:25].to(device)
utils.save_image(vis_inputs, pic_dir + '/inputs.png', nrow=5, normalize=True)
save_name = os.path.join(result_dir,'model.pt')
early_stop = EarlyStop(patience=20, save_name=save_name)

gms_loss_list = []
ssim_loss_list = []
l2_loss_list = []
val_loss_list = []
for epoch in range(1, args.epochs + 1):
    if epoch == 250:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    loop = tqdm(train_loader, unit='batch', desc='Train [Epoch {:>3}]'.format(epoch))
    net.train()
    scaler = torch.cuda.amp.GradScaler()

    l2_losses = []
    gms_losses = []
    ssim_losses = []
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()

    for i, (data, _, _) in enumerate(loop):
        optimizer.zero_grad()
        data = data.to(device)

        #k_value = random.sample(args.k_value, 1)
        k_value = [args.k_value[np.random.randint(0, len(args.k_value))]]
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms = next(Ms_generator)

        inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        outputs = [net(x) for x in inputs]
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))

        gms_loss = msgms(data, output)
        ssim_loss = ssim(data, output)
        l2_loss = mse(data, output)
        
        loss = args.lambda_G * gms_loss + args.lambda_S * ssim_loss + l2_loss

        loss.backward()
        optimizer.step()

        gms_losses.append(gms_loss.item())
        ssim_losses.append(ssim_loss.item())
        l2_losses.append(l2_loss.item())

    gms_loss_list.append(np.average(gms_losses))
    ssim_loss_list.append(np.average(ssim_losses))
    l2_loss_list.append(np.average(l2_losses))
    logger.info('[Train Epoch {}/{}] GMS_Loss : {:.6f} SSIM_Loss: {:.6f} L2_Loss: {:.6f}'.format(epoch, args.epochs, gms_loss_list[-1], ssim_loss_list[-1], l2_loss_list[-1]))

    if epoch % 10 == 0:
        utils.save_image(data, pic_dir + '/train{}_input.png'.format(epoch), normalize=True)
        utils.save_image(output, pic_dir + '/train{}_output.png'.format(epoch), normalize=True)

    v_loss = []
    with torch.no_grad():
        net.eval()
        loop_val = tqdm(val_loader, unit='batch', desc='Val   [Epoch {:>3}]'.format(epoch))

        for i, (data, _, _) in enumerate(loop_val):
            data = data.to(device)

            #k_value = random.sample(args.k_value, 1)
            k_value = [args.k_value[np.random.randint(0, len(args.k_value))]]
            Ms_generator = gen_mask(k_value, 3, args.img_size)
            Ms = next(Ms_generator)

            inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
            outputs = [net(x) for x in inputs]
            output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
            
            gms_loss = msgms(data, output)
            ssim_loss = ssim(data, output)
            l2_loss = mse(data, output)

            loss = args.lambda_G * gms_loss + args.lambda_S * ssim_loss + l2_loss

            v_loss.append(loss.item())

        val_loss_list.append(np.average(v_loss))
        logger.info('[Val Epoch {}/{}] Val Loss : {:.6f}'.format(epoch, args.epochs, val_loss_list[-1]))

    if epoch % 10 == 0:
        with torch.no_grad():
            inputs = [vis_inputs * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
            outputs = [net(x) for x in inputs]
            output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))
            utils.save_image(output, pic_dir + '/validation-{}.png'.format(epoch), nrow=5, normalize=True)
            logger.info('Validation picture {} exported.'.format(epoch))

    if (early_stop(val_loss=val_loss_list[-1], model=net, optimizer=optimizer)):
        logger.info('Earlystop at {}'.format(epoch))
        break

logger.info('Train picture exported.')

#np.save(result_dir + '/train_loss_list.npy', np.array(train_loss_list))
#torch.save(net.state_dict(), result_dir+'/ClothRIAD_{}.model'.format(epoch))
#logger.info('Model exported.')

# output loss_img
fig = plt.figure(figsize=(6,6))
#train_loss_list = np.load(result_dir + '/train_loss_list.npy')
plt.plot(gms_loss_list, label='GMS_loss')
plt.plot(ssim_loss_list, label='SSIM_loss')
plt.plot(l2_loss_list, label='L2_loss')
plt.plot(val_loss_list, label='Val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 0.6])
plt.grid()
fig.savefig(result_dir + '/loss.png')
logger.info('Loss Graph exported.')

with open(os.path.join(result_dir, 'train.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['data', 'class', 'size', 'Epoch', 'batch', 'val_rate', 'lr', 'init', 'seed', 'lam_G', 'lam_S', 'k_val', 'GMS', 'SSIM', 'L2', 'v_loss'])
    writer.writerow([args.dataset, args.data_type, args.img_size, args.epochs, args.batch_size, args.val_rate, args.lr, args.init_type, args.seed, args.lambda_G, args.lambda_S, args.k_value, gms_loss_list[-1], ssim_loss_list[-1], l2_loss_list[-1], val_loss_list[-1]])

logger.info('{} Completed!'.format(result_dir))