import os
import torch
import torch.nn.functional as F
import  math
import numpy as np
import torch.nn as nn
from datetime import datetime
from torchvision.utils import make_grid
from scipy.stats import spearmanr
# from SFFFF.DMRA.fusion import DMRA
# from sunfanNet.MENet_VGG16_Res34_4 import   SFNet
# from Convnext.CONVNEXT_1 import SFNet_Conv_tiny4
from Fourth.ACCoNet_wave_models import ACCoNet_Res_student
from Fourth.SwinNet.models.swinNet import SwinNet
# from Convnext.swinNet import SwinNet
from rgbt_dataset import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from config import opt
from torch.cuda import amp
import random
from Fourth.ACCoNet_main import pytorch_iou
# set the device for training
cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# build the model

model = ACCoNet_Res_student()
model2 = SwinNet()
# print('NOW USING:SSFFF5_NEWVMMM_RES34')
# if (opt.load is not None):
# model.load_state_dict(torch.load('/media/sunfan/date/Paper_4/WaveMLP_M.pth'),strict=False)

# checkpoint = torch.load('/media/sunfan/date/Paper_4/WaveMLP_M.pth', map_location='cpu')  # 加载模型文件，pt, pth 文件都可以；
# if torch.cuda.device_count() > 1:
#     # 如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"。
#     model = nn.DataParallel(model)
# model.load_state_dict(checkpoint,strict=False) # 接着就可以将模型参数load进模型。
#
# model.load_pre("/media/sunfan/date/Paper_4/WaveMLP_M.pth")
model.load_pre()
# model2.load_state_dict(torch.load("/media/sunfan/date/Paper_4/SwinNet/SwinTransNet_epoch_best.pth"))
# model.load_pre("/home/sunfan/Downloads/Pretrain/swin_base_patch4_window12_384_22kto1k.pth")
# model.load_pre("/home/sunfan/Downloads/Pretrain/swin_tiny_patch4_window7_224.pth")
# print('load model from ', opt.load)
# device = torch.cuda.set_device(0)
# model.to(device)
model.cuda()
# model2.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
train_dataset_path = opt.lr_train_root
image_root = train_dataset_path + '/RGB/'
depth_root = train_dataset_path + '/T/'
gt_root = train_dataset_path + '/GT/'
bound_root = train_dataset_path + '/bound/'
val_dataset_path = opt.lr_val_root
val_image_root = val_dataset_path + '/RGB/'
val_depth_root = val_dataset_path + '/T/'
val_gt_root = val_dataset_path + '/GT/'
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')

train_loader = get_loader(image_root, gt_root,depth_root,bound_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# print(len(train_loader))
test_loader = test_dataset(val_image_root, val_gt_root,val_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info(save_path + "Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
import torch.nn as nn


def calc_corr(a, b):

    a_avg = a.mean()
    b_avg = b.mean()

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = ((a-a_avg) * (b - b_avg)).sum()
    # cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(((a-a_avg) ** 2).sum() * ((b-b_avg) ** 2).sum())
    # sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq
    return corr_factor
def compute_rank_correlation(att, grad_att):
    """
    Function that measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """
    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

    att = att.sort(dim=1)[1]
    grad_att = grad_att.sort(dim=1)[1]
    correlation = _rank_correlation_(att.float(), grad_att.float())
    return correlation

class Corr_BCEloss(nn.Module):
    def __init__(self):
        super(Corr_BCEloss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()
        # self.relu = nn.ReLU(inplace=True)
    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.shape
        losses = []


        for inputs, targets in zip(input_scale, taeget_scale):
            predi = torch.sigmoid(inputs)
            inter = (predi * targets).sum(dim = (1, 2))
            union = (predi + targets).sum(dim = (1, 2))
            iou = (inter+1) / (union - inter+1)
            # print('iou',iou)

            # weight =  compute_rank_correlation(predi,targets)

            # print('weiht',weight)
            # print('weight',weight.type)
            # weight=ch(weight)
            lossall = iou + self.nll_lose(inputs, targets)#3
            # lossall = weight + self.nll_lose(inputs, targets)  # 3
            # lossall = (1-weight) * self.nll_lose(inputs, targets)#2
            # print(targets.shape)
            # lossall = sigmoid_cross_entropy_loss(inputs,targets)
            losses.append(lossall)

        total_loss = sum(losses)/b

        return total_loss


class WBCE2(nn.Module):
    def __init__(self):
        super(WBCE2, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        b, _, _, _ = input_scale.size()
        for inputs, targets in zip(input_scale, taeget_scale):
            res = torch.sigmoid(inputs)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            weight = torch.abs(res - targets)
            # print(torch.sum(weight))
            weight = torch.sum(weight) / torch.sum(targets)
            # print(torch.sum(targets))
            lossall = weight  * self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses) / b
        return total_loss



def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

class Triangle_loss(nn.Module):
    def __init__(self):
        super(Triangle_loss, self).__init__()
        self.nll_lose = nn.BCEWithLogitsLoss()

    def forward(self, input_scale, taeget_scale):
        b,_,_,_ = input_scale.size()
        loss = []
        for inputs, targets in zip(input_scale, taeget_scale):
            pred = torch.sigmoid(inputs)
            weight = torch.abs(pred - targets)
            weight = torch.sum(weight) / torch.sum(targets)
            inter = (pred * targets).sum(dim=(1, 2))
            union = (pred + targets).sum(dim=(1, 2))
            IOU = inter / (union - inter)
            BCE = self.nll_lose(inputs, targets)
            losses = (weight * BCE) / (IOU + 1)
            loss.append(losses)
        total_loss = sum(loss)
        return total_loss / b
WCE = Triangle_loss().cuda()

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
# criterion = cross_entropy2d()
# criterion = WCE()
criterion = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.BCELoss()
# Trianglecriterion = Triangle_loss().cuda()
# L1criterion = nn.SmoothL1Loss().cuda()
# Depthcriterion = Depth_loss().cuda()


# 超参数
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
Sacler = amp.GradScaler()

# train function
length = 821

def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts,depths,bound) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            ima = images
            dep = depths
            images = images.cuda()
            # print(images.shape)
            depths = depths.cuda()
            gts = gts.cuda()
            bound = bound.cuda()
            _,_,w_t,h_t = gts.size()
            # gts = gts.data.cpu().numpy()
            gts2 = F.upsample(gts, (w_t // 2, h_t // 2), mode='bilinear')
            gts3 = F.upsample(gts, (w_t // 4, h_t // 4), mode='bilinear')
            gts4 = F.upsample(gts, (w_t // 8, h_t // 8), mode='bilinear')
            gts5 = F.upsample(gts, (w_t // 16, h_t // 16), mode='bilinear')
            # Gabor_ls变为三通道,Gabor_rs变为三通道
            # n, c, h, w = images.size()  # batch_size, channels, height, weight
            # Gabor_ls = Gabor_ls.view(n, h, w, 1).repeat(1, 1, 1, c)
            # Gabor_ls = Gabor_ls.transpose(3, 1)
            # Gabor_ls = Gabor_ls.transpose(3, 2)
            #
            s1, s2, s3, s4, s1_sig, s2_sig, s3_sig, s4_sig= model(images, depths)
            # target,_ = model2(ima, dep)
            # target = target.cuda()

            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
            # loss5 = CE(s5, gts) + IOU(s5_sig, gts)
            # loss5 = CE(s1, target) + IOU(s1_sig, target)
            # loss6 = CE(s2, target) + IOU(s2_sig, target)
            # loss7 = CE(s3, target) + IOU(s3_sig, target)
            # loss8 = CE(s4, target) + IOU(s4_sig, target)
            loss = loss1 + loss2 + loss3 + loss4 #+ loss5 +loss6 + loss7 + loss8

            loss.backward()
            # with amp.autocast():
            # n, c, h, w = images.size()
            # depths = depths.view(n, h, w, 1).repeat(1, 1, 1, c)
            # depths = depths.transpose(3, 1)
            # depths = depths.transpose(3, 2)
            # print('depths',depths.shape)
            #out = model(images, depths)
            # out = torch.softmax(out,dim=1)

            # out = out[0][1]
            # print(out)
            # out = torch.sigmoid(out)
            # print(out)
            # out0 = torch.sigmoid(out[0])
            # out1 = torch.sigmoid(out[1])
            # out2 = torch.sigmoid(out[2])
            # out3 = torch.sigmoid(out[3])
            # out4 = torch.sigmoid(out[4])
            # out5 = torch.sigmoid(out[5])
            # out6 = torch.sigmoid(out[6])
            # out7 = torch.sigmoid(out[7])
            # out8 = torch.sigmoid(out[8])
            # out9 = torch.sigmoid(out[9])

            # out0 = out[0]
            # out0 = out[0]
            # out1 = out[1]
            # out2 = out[2]
            # out3 = out[3]
            # out4 = out[4]
            # out5 = out[5]
            # out6 = out[6]
            # out7 = out[7]
            # out8 = out[8]
            # out9 = out[9]
            # loss = criterion(out, gts)
            # print(loss)
            # print('out0',out0.shape)
            # print('gts', gts.shape)WCE
            # # loss2 = joint_loss(out, gts).cuda()
            # loss2 = joint_loss(out0, gts).cuda()
            # loss3 = joint_loss(out1, gts).cuda()
            # loss4 = joint_loss(out2, gts).cuda()
            # loss5 = joint_loss(out3, gts).cuda()
            # loss6 = joint_loss(out4, gts).cuda()
            # loss7 = joint_loss(out5, bound).cuda()


            # loss = loss2 #+ loss3 + loss4+ loss5 + loss6 #+ loss7 #+loss8 + loss9 + loss10 +loss11
            # print(loss2,loss7)
            # loss.backward()
            # Sacler.scale(loss).backward()
            # clip_gradient(optimizer, opt.clip)
            # Sacler.step(optimizer)
            # Sacler.update()
            optimizer.step()
            step += 1
            epoch_step = epoch_step +1
            loss_all =loss_all + loss.item()
            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}],W*H [{:03d}*{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch+1, opt.epoch, w_t, h_t, i, total_step, loss.item()))
                # print('{} ,W*H [{:03d}*{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                #       format(datetime.now(), w_t, h_t, i, total_step, loss.item()))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch+1, opt.epoch, i, total_step, loss.item()))
                writer.add_scalar('Loss/total_loss', loss, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data,1,normalize=True)
                writer.add_image('train/RGB',grid_image, step)
                grid_image = make_grid(depths[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('train/Ti', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('train/gt', grid_image, step)
                # res = out0[0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/last_out', torch.tensor(res), step, dataformats='HW')
                #
                # res = out5[0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/bound', torch.tensor(res), step, dataformats='HW')
                #
                # res = out[2][0].clone().sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/out2', torch.tensor(res), step, dataformats='HW')
                #
                # res = out[3][0].clone().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('SOD_contrast/out1', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch+1, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch+1) % 50 == 0 or (epoch+1) == opt.epoch:
            torch.save(model.state_dict(), save_path + 'RES34_1_epoch_{}_test.pth'.format(epoch+1))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'RES34_1_epoch_{}_test.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

#

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            # print(image,right,name,Gabor_l,Gabor_r)
            gt = gt.cuda()
            image = image.cuda()
            depth = depth.cuda()

            # with amp.autocast():
            res = model(image, depth)
            # res = torch.sigmoid(res)
            res = torch.sigmoid(res[0])
            # res = res[0]
            # res = torch.sigmoid(res)
            res = (res-res.min())/(res.max()-res.min()+1e-8)
            mae_train = torch.sum(torch.abs(res - gt)) / (224.0 * 224.0)
            # mae_train =torch.sum(torch.abs(res-gt))*1.0/(torch.numel(gt))
            mae_sum = mae_train.item()+mae_sum
            #mae_sum += torch.sum(torch.abs(res - gt)) / torch.numel(gt)
                # print(torch.numel(gt))
                # print(mae_sum)
        # mae = mae_sum / test_loader.size  mae / length:.8f
        mae = mae_sum /length
        # print(mae,test_loader.size)
        #writer.add_scalar('MAE', torch.as_tensor(mae), global_step=epoch)
        # res = res[0].clone().data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # writer.add_image('SOD_contrast/test_predict', torch.tensor(res), step, dataformats='HW')
        # print(' MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(mae, best_mae, best_epoch))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'RES34_1_best_mae_test.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))



if __name__ == '__main__':
    print("Start train...")
    start_time = datetime.now()
    for epoch in range(opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
    finish_time = datetime.now()
    h, remainder = divmod((finish_time - start_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time)