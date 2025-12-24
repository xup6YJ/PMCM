import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from utils.create import get_task_name
from utils.util import plot_grid, plot_overlap, plot_KDE, plot_DST
from dataloaders.dataset import *
from networks.net_factory import net_factory
from networks.map import MappingNet, MappingNet_single_task
from utils.losses import MutualBoundaryLoss, TaskPredConsistency, kl_loss, compute_sdf
from torchvision.utils import make_grid

'''
Base on mean 
+simce
data fixed

'''
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/bspubuntu/1TBSSD/A_exp/dataset', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset_name', type=str, choices=["LA", "Pancreas_CT"], default='LA', help='dataset')               # todo change dataset path

parser.add_argument('--exp', type=str, default='MCNet', help='exp_name')
parser.add_argument('--model', type=str, default='mine3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=6000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--nramp', action='store_true', default=False)


parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--n_class', type=int,  default=2)
parser.add_argument('--l_loss', type=str, choices=["org", "cd"], default='org')

parser.add_argument('--bd_loss', action = 'store_true',  default=True, help='boundary Loss')
parser.add_argument('--bd_loss_weight', type=float, default=1.0)
parser.add_argument('--bd_channel', type=int, default=1)
parser.add_argument('--bd_class', type=str,  choices=["all", "class"], default='class')
parser.add_argument('--bd_entropy_weighted', action = 'store_true',  default=False)
parser.add_argument('--bd_entropy_weighted_one', action = 'store_true',  default=False)


parser.add_argument('--task_cons', action = 'store_true',  default=True)
parser.add_argument('--task_net_pred', action='store_true', default=True)
parser.add_argument('--task_net_detach', action='store_true', default=True)
parser.add_argument('--task_net_siamese_map', action='store_true', default=True, help='V&R same maaping net')
parser.add_argument('--task_net_map_pred', action='store_true', default=False, help='pred SDM using mapping net')
parser.add_argument('--task_ent', action='store_true', default=False)
parser.add_argument('--task_ent_det', action='store_true', default=False)
parser.add_argument('--task_lca', action='store_true', default=False)
parser.add_argument('--task_prot', action='store_true', default=False)
parser.add_argument('--task_c', action = 'store_true',  default=False, help='boundary Loss')

parser.add_argument('--task_exp', action = 'store_true',  default=False)
parser.add_argument('--task_exp_type', type=str,  choices=["dtc", "single"], default='single', help='same mapping net or not')

parser.add_argument('--task_cons_weight', type=float, default=0.1)
parser.add_argument('--task_bd_seg_weight', type=float, default=1)
parser.add_argument('--task_net_mode', type=str,  choices=["ind", "dep"], default='ind', help='same mapping net or not')
parser.add_argument('--task_criteria', type=str,  default="kl")
parser.add_argument('--task_update', type=str,   default='own')
parser.add_argument('--task_ent_mode', type=str,  choices=["erl", "cor", 'none'], default='none')
parser.add_argument('--task_unc_pse', type=int, default=2, help='number of pseudo label')

parser.add_argument('--task_ag', action='store_true', default=False)
parser.add_argument('--task_ag_ent', action='store_true', default=False)
parser.add_argument('--task_ag_ent_det', action='store_true', default=False)
parser.add_argument('--task_ag_erl', action='store_true', default=False)
parser.add_argument('--task_ag_entmin', action='store_true', default=False)
parser.add_argument('--task_ag_amb', action='store_true', default=False)
parser.add_argument('--task_ag_sharp', action='store_true', default=False)

parser.add_argument('--task_ag_criteria', type=str,  choices=["mse", "ce"], default='mse')
parser.add_argument('--task_ag_ans', type=str,  choices=["mean", "unc"], default='unc')
parser.add_argument('--task_ag_ent_mode', type=str,  choices=["norm", "exp"], default='norm')
parser.add_argument('--task_ag_entmin_mode', type=str,  choices=["dep", "ind"], default='ind')
parser.add_argument('--task_ag_mode', type=str,  choices=["seg", "bd", "both"], default='both')
parser.add_argument('--task_ag_weight', type=float, default=0.1)
parser.add_argument('--task_ag_pse', type=int, default=4, help='pseudo label prediction number')
parser.add_argument('--task_lca_mode', type=str,  choices=["erl", "disagh"], default='erl')
parser.add_argument('--task_lca_ent_mode', type=str,  choices=["rw", "org"], default='org')
parser.add_argument('--task_ag_erl_mode', type=str,  choices=["unc", "org"], default='org')
parser.add_argument('--task_ag_seg_weight', type=float, default=1)
parser.add_argument('--task_ag_bd_weight', type=float, default=1)

parser.add_argument('--mapping_loss', action='store_true', default=True)
parser.add_argument('--mapping_det', action='store_true', default=False)
parser.add_argument('--mapping_ans_det', action='store_true', default=True)
parser.add_argument('--mapping_id', action='store_true', default=False)
parser.add_argument('--mapping_ramp', action='store_true', default=True)
# parser.add_argument('--mapping_det_mode', type=str,  choices=["in", "out", "all"], default=None)
parser.add_argument('--mapping_loss_weight', type=float, default=0.005)
parser.add_argument('--mapping_bd_seg_weight', type=float, default=1)
parser.add_argument('--mapping_dim', type=int, default=8)
parser.add_argument('--mapping_loss_type', type=str,  default='kl')
parser.add_argument('--mapping_id_weight', type=float, default=0.1)


parser.add_argument('--prot', action='store_true', default=True)
parser.add_argument('--prot_cons', action='store_true', default=True)
parser.add_argument('--prot_amb', action='store_true', default=False)
# parser.add_argument('--prot_bd', action='store_true', default=False)
# parser.add_argument('--prot_sim', action='store_true', default=False)
parser.add_argument('--prot_fusion', action='store_true', default=True)
# parser.add_argument('--prot_unc', action='store_true', default=False)
# parser.add_argument('--prot_soft', action='store_true', default=False)
# parser.add_argument('--prot_lb', action='store_true', default=False, help='label guided prot')
# parser.add_argument('--prot_task', action='store_true', default=False)
parser.add_argument('--prot_un', action='store_true', default=False)
parser.add_argument('--prot_ual', action='store_true', default=True)
# parser.add_argument('--prot_ual_det', action='store_true', default=False)
parser.add_argument('--prot_ual_d', action='store_true', default=True)
# parser.add_argument('--prot_bank_alt', action='store_true', default=False)
parser.add_argument('--prot_ual_weight', type=float, default=1.0)

parser.add_argument('--prot_mode', type=str,  default='org')
# parser.add_argument('--prot_cal_mode', type=str,  choices=["lw", "org"], default='org')
parser.add_argument('--prot_data', type=str,   default='all')
parser.add_argument('--prot_bank_data', type=str,   default='all')
parser.add_argument('--prot_amb_mode', type=str,   default='org')

# parser.add_argument('--prot_bd_mode', type=str,  choices=["map", "mean", "test"], default='map')
# parser.add_argument('--prot_bd_w', type=str,  choices=["org",  "add", 'addu', 'addud', 'ind', 'indu', 'indur'], default='add')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
# parser.add_argument('--prot_bd_data', type=str,  choices=["all", "unsup"], default='all')

parser.add_argument('--feat_sim', action='store_true', default=True)
parser.add_argument('--feat_sim_weight', type=float, default=1.0)

parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--plot_kde', action='store_true', default=False)
parser.add_argument('--val', action='store_true', default=False)

args = parser.parse_args()


# args.exp = 'MR_flod' + str(args.cv)

# model_name = get_task_name(args)
model_name = f'PMCM_{args.dataset_name}_{args.labelnum}_{args.max_iteration}'
print('model name: ', model_name)
train_data_path = args.root_path
snapshot_path = "../"+model_name+"/" 


num_classes = args.n_class
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    # args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    # args.root_path = args.root_path + 'data/Pancreas/'
    args.max_samples = 62
train_data_path = args.root_path


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr


def get_current_consistency_weight(weight, epoch):
    if args.nramp:
        epoch =  epoch // (args.max_iteration/6000)
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)



if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


    model = net_factory(args=args, net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    if args.task_cons:
        seg_bd_map = MappingNet_single_task(args).cuda()
        bd_seg_map = MappingNet_single_task(args).cuda()

    train_list = 'train.list'    # todo change training flod
    if args.dataset_name == "LA":
        db_train = LAHeartD(args,
                            base_dir=train_data_path,
                           split='train',
                           train_flod=train_list,                   # todo change training flod
                           transform=transforms.Compose([
                               RandomRotFlipD(with_sdf = args.bd_loss),
                               RandomCrop(patch_size, with_sdf = args.bd_loss),
                               ToTensor(with_sdf = args.bd_loss),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = PancreasD(args,
                             base_dir=train_data_path,
                            split='train',
                            train_flod=train_list,                   # todo change training flod
                            transform=transforms.Compose([
                                RandomCrop(patch_size, with_sdf = args.bd_loss),
                                ToTensor(with_sdf = args.bd_loss),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=4,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.task_cons:
        params = list(model.parameters()) + list(seg_bd_map.parameters()) + list(bd_seg_map.parameters())
        optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0001)

    else:
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter("../log/"+model_name )
    logging.info("{} itertations per epoch".format(len(trainloader)))
    pixel_criterion = losses.ce_loss_mask
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.Binary_dice_loss
    mse_loss = nn.MSELoss(reduction='none')

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    
    if args.task_cons:
        seg_bd_map.train()
        bd_seg_map.train()

    early_stop = 0
    directly_stop = False

    time1 = time.time()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            if args.bd_loss:
                volume_batch, label_batch, sdf = sampled_batch['image'], sampled_batch['label'], sampled_batch['sdf']
                volume_batch, label_batch, sdf = volume_batch.cuda(), label_batch.cuda(), sdf.cuda()
            
            else:
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            loss = 0
            grid = {}

            model.train()
            # outputs, outfeats = model(volume_batch)
            all_output =  model(volume_batch)
            outputs = all_output['seg']
            outfeats = all_output['prot']
            if args.bd_loss:
                bd_out1 = all_output['bd'][0]
                bd_out2 = all_output['bd'][1]
            if args.simsiam:
                prot_pred_out1 = all_output['prot_pred'][0] #input 1 should be supervised by 2
                prot_pred_out2 = all_output['prot_pred'][1]
                feat1_ans = outfeats[1].detach()
                feat2_ans = outfeats[0].detach()

                criterion = nn.CosineSimilarity(dim=1)
                sim_loss = -(criterion(prot_pred_out1, feat1_ans).mean() + criterion(prot_pred_out2, feat2_ans).mean()) * 0.5
                loss += sim_loss*args.sim_weight

            seg_out1 = outputs[0]
            seg_out2 = outputs[1]
            seg_out_soft1 = F.softmax(seg_out1, dim=1)
            seg_out_soft2 = F.softmax(seg_out2, dim=1)

            num_outputs = len(outputs)

            y_all = torch.zeros((num_outputs, ) + outputs[0].shape).cuda()

            #loss
            loss_seg = 0
            loss_seg_dice = 0
            loss_cs = 0
            uloss_cs = 0
            u_sim_loss = 0

            for idx in range(num_outputs):
                y = outputs[idx]
                y_prob = F.softmax(y, dim=1)
                if args.l_loss == 'cd':
                    loss_seg += F.cross_entropy(y_prob[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)

                y_all[idx] = y_prob

            loss_consist = 0
            pixel_consist = 0
            loss += args.lamda * loss_seg_dice
            if args.l_loss == 'cd':
                loss += args.lamda * loss_seg
                writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)

            feat_sim = F.cosine_similarity(outfeats[0].detach(), outfeats[1].detach(), dim=1).mean()
            feat_map1 = torch.mean(outfeats[0].detach(), dim = 1)
            feat_map2 = torch.mean(outfeats[1].detach(), dim = 1)
            #normalize map
            feat_map1n = (feat_map1 - feat_map1.min())/(feat_map1.max() - feat_map1.min())
            feat_map2n = (feat_map2 - feat_map2.min())/(feat_map2.max() - feat_map2.min())

            # plot grid
            if args.bd_loss:
                dis_to_mask1 = torch.sigmoid(-1500*F.tanh(bd_out1))
                dis_to_mask2 = torch.sigmoid(-1500*F.tanh(bd_out2))

            seg_pred1 = seg_out_soft1.argmax(1)[0]
            seg_pred2 = seg_out_soft2.argmax(1)[0]


            writer.add_scalar('Co_loss/sim2map', feat_sim , iter_num)

            if args.bd_loss:
                BD_loss = MutualBoundaryLoss(args)
                bd_loss, grid = BD_loss(seg_out1, bd_out1, seg_out2, bd_out2, labeled_bs, label_batch, sdf = sdf, input = volume_batch, grid = grid)
                v_bd_loss = bd_loss['bd_loss_1']
                r_bd_loss = bd_loss['bd_loss_2']

                bda_loss = args.lamda * (v_bd_loss + r_bd_loss)
                loss += args.bd_loss_weight* bda_loss
                writer.add_scalar('Labeled_loss/bd_loss', bda_loss, iter_num)

            # unsuperivsed loss 
            if args.task_exp:
                task_weight = get_current_consistency_weight(weight= args.task_cons_weight, epoch = iter_num//150)
                if args.task_exp_type == 'dtc':
                    dis_to_mask1 = torch.sigmoid(-1500*F.tanh(bd_out1))
                    dis_to_mask2 = torch.sigmoid(-1500*F.tanh(bd_out2))

                    # dis_to_mask = torch.sigmoid(-1500*outputs_tanh)
                    consistency_loss = torch.mean((dis_to_mask1 - seg_out_soft1) ** 2) + torch.mean((dis_to_mask2 - seg_out_soft2) ** 2)
                    loss += task_weight* consistency_loss

                elif args.task_exp_type == 'single':
                    with torch.no_grad():
                        seg_out_pred1 = seg_out1.argmax(1)
                        seg_out_pred2 = seg_out2.argmax(1)
                        sdf_pred1 = compute_sdf(seg_out_pred1.cpu().numpy(), seg_out_soft1[:, 0, ...].shape)
                        sdf_pred1 = torch.from_numpy(sdf_pred1).float().cuda()
                        sdf_pred2 = compute_sdf(seg_out_pred2.cpu().numpy(), seg_out2[:, 0, ...].shape)
                        sdf_pred2 = torch.from_numpy(sdf_pred2).float().cuda()

                    consistency_loss = torch.mean( (sdf_pred1 - F.tanh(bd_out1[:,1,...])) ** 2) + torch.mean(  (sdf_pred2 - F.tanh(bd_out1[:,1,...]) ** 2))
                    loss += task_weight* consistency_loss

                
            if args.task_cons:
                task_weight = get_current_consistency_weight(weight= args.task_cons_weight, epoch = iter_num//150)
                task_cons = TaskPredConsistency(args)

                if args.task_net_pred:
                    # using mapping net to pred task
                    seg_bd_pred_1 = seg_bd_map(seg_out1, input_type = 'seg')['seg_bd_out']
                    seg_bd_pred_2 = seg_bd_map(seg_out2, input_type = 'seg')['seg_bd_out']
                    bd_seg_pred_1 = bd_seg_map(bd_out1, input_type = 'bd')['bd_seg_out']
                    bd_seg_pred_2 = bd_seg_map(bd_out2, input_type = 'bd')['bd_seg_out']

                    bd_seg_pred_soft_1 = F.softmax(bd_seg_pred_1, dim=1)
                    bd_seg_pred_soft_2 = F.softmax(bd_seg_pred_2, dim=1)

                    # boundary plot refinement
                    v_b_tanR = torch.abs(seg_bd_pred_1)
                    r_b_tanR = torch.abs(seg_bd_pred_2)

                
                if args.task_update =='own':
                    assert bd_seg_pred_1.shape == seg_out_soft1.shape == seg_bd_pred_1.shape

                    v_task_loss = task_cons(bd_seg_pred_1, seg_out_soft1, seg_bd_pred_1, F.tanh(bd_out1))
                    v_bd_seg_loss = v_task_loss['bd_seg_loss']*args.task_bd_seg_weight
                    v_seg_bd_loss = v_task_loss['seg_bd_loss']

                    r_task_loss = task_cons(bd_seg_pred_2, seg_out_soft2, seg_bd_pred_2, F.tanh(bd_out2))
                    r_bd_seg_loss = r_task_loss['bd_seg_loss']*args.task_bd_seg_weight
                    r_seg_bd_loss = r_task_loss['seg_bd_loss']


                loss += task_weight* (v_bd_seg_loss + v_seg_bd_loss + r_bd_seg_loss + r_seg_bd_loss)

                writer.add_scalar('weight/task_weight', task_weight, iter_num)
                writer.add_scalar('loss/1_bd_seg_loss', v_bd_seg_loss, iter_num)
                writer.add_scalar('loss/1_seg_bd_loss', v_seg_bd_loss, iter_num)
                writer.add_scalar('loss/2_bd_seg_loss', r_bd_seg_loss, iter_num)
                writer.add_scalar('loss/2_seg_bd_loss', r_seg_bd_loss, iter_num)

            if args.feat_sim:
                feat_sim0 = 1 + F.cosine_similarity(outfeats[0], outfeats[1].detach(), dim=1).mean()
                feat_sim1 = 1 + F.cosine_similarity(outfeats[1], outfeats[0].detach(), dim=1).mean()
                f_weight = get_current_consistency_weight(args.feat_sim_weight, iter_num // 150)
                loss += f_weight * (feat_sim0 + feat_sim1)/2
                writer.add_scalar('Co_loss/feat_sim', (feat_sim0 + feat_sim1)/2, iter_num)
                writer.add_scalar('weight/feat_sim_weight', f_weight, iter_num)
                

            if args.prot:
                if args.prot_mode == 'org':
                    p_weight = get_current_consistency_weight(1, iter_num // 150)
                    writer.add_scalar('weight/p_weight', p_weight, iter_num)
                    simce_weight = get_current_consistency_weight(args.sim_uramp_weight, iter_num // 150)
                    writer.add_scalar('weight/sim_uramp_weight', simce_weight, iter_num)
                    for i in range(num_outputs):
                        for j in range(num_outputs):
                            if i != j:
                                uncertainty_o1 = -1.0 * torch.sum(y_all[i] * torch.log(y_all[i] + 1e-6), dim=1)
                                uncertainty_o2 = -1.0 * torch.sum(y_all[j] * torch.log(y_all[j] + 1e-6), dim=1)
                                mask = (uncertainty_o1 > uncertainty_o2).float()

                                batch_o, c_o, w_o, h_o, d_o = y_all[j].shape
                                batch_f, c_f, w_f, h_f, d_f = outfeats[j].shape

                                teacher_o = y_all[j].reshape(batch_o, c_o, -1)
                                teacher_f = outfeats[j].reshape(batch_f, c_f, -1)
                                stu_f = outfeats[i].reshape(batch_f, c_f, -1)


                                if args.prot_ual:
                                    if args.prot_bank_alt:
                                        prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()
                                        index = torch.argmax(y_all[j], dim=1, keepdim=True)

                                        for ba in range(batch_f):
                                            for n_class in range(num_classes):
                                                mask_temp = (index[ba] == n_class).float()
                                                top_fea = outfeats[j][ba] * mask_temp
                                                prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)
                                        
                                        label_prot_bank = torch.mean(prototype_bank[:labeled_bs], dim=0)
                                        unlabel_prot_bank = torch.mean(prototype_bank[labeled_bs:], dim=0)
                                        # unlabel_prot_bank align with label_prot_bank
                                        unlabel_prot_bank = unlabel_prot_bank.contiguous().view(c_o, -1)
                                        label_prot_bank = label_prot_bank.contiguous().view(c_o, -1)
                                        if args.prot_ual_det:
                                            label_prot_bank = label_prot_bank.detach()
                                        
                                        if not args.prot_ual_d:
                                            # positive
                                            ul_cos = 0
                                            for n_class in range(num_classes):
                                                ul_cosd = 1-F.cosine_similarity(unlabel_prot_bank[n_class], label_prot_bank[n_class], dim=0)
                                                # print(ul_cosd)
                                                ul_cos += ul_cosd

                                            u_sim_loss += ul_cos / num_classes

                                        else:
                                            pos_similarities = []
                                            neg_similarities = []

                                            # Calculate positive and negative similarities for each class
                                            for c1 in range(num_classes):
                                                # Positive similarity (unlabeled i-th vector with labeled i-th vector)
                                                pos_sim = F.cosine_similarity(unlabel_prot_bank[c1], label_prot_bank[c1], dim=0)
                                                pos_similarities.append(pos_sim.unsqueeze(0))
                                                
                                                # Loop through all other classes for negative pairs
                                                for c2 in range(num_classes):
                                                    if c1 != c2:
                                                        # Negative similarity (unlabeled i-th vector with labeled j-th vector)
                                                        neg_sim = F.cosine_similarity(unlabel_prot_bank[c2], label_prot_bank[c2], dim=0)
                                                        neg_similarities.append(neg_sim.unsqueeze(0))

                                            # Stack all positive and negative similarities
                                            pos_similarities = torch.cat(pos_similarities, dim=0).unsqueeze(0)
                                            neg_similarities = torch.cat(neg_similarities, dim=0).unsqueeze(0)

                                            # Combine positive and negative similarities into a similarity matrix
                                            sim_matrix = torch.cat([pos_similarities, neg_similarities], dim=0)

                                            # Create labels: 0 for positive pairs, 1 for negative pairs
                                            labels = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=sim_matrix.device)

                                            # Compute cross-entropy loss
                                            u_sim_loss += F.cross_entropy(sim_matrix, labels) / sim_matrix.shape[0]


                                prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()
                                if args.prot_bank_alt:
                                    if iter_num % 2 == 0:
                                        index = torch.argmax(y_all[i], dim=1, keepdim=True)
                                        on = i
                                    else:
                                        index = torch.argmax(y_all[j], dim=1, keepdim=True)
                                        on = j

                                    for ba in range(batch_f):
                                        for n_class in range(num_classes):
                                            mask_temp = (index[ba] == n_class).float()
                                            top_fea = outfeats[on][ba] * mask_temp
                                            prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)
                                else:
                                    index = torch.argmax(y_all[j], dim=1, keepdim=True)

                                    for ba in range(batch_f):
                                        for n_class in range(num_classes):
                                            mask_temp = (index[ba] == n_class).float()
                                            top_fea = outfeats[j][ba] * mask_temp
                                            prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)

                                if args.prot_bank_data == 'ind':
                                    prototype_bank = F.normalize(prototype_bank, dim=-1)
                                    mask_t = torch.zeros_like(y_all[i]).cuda()
                                    for ba in range(batch_o):
                                        for n_class in range(num_classes):
                                            class_prototype = prototype_bank[ba, n_class]
                                            mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                                    class_prototype.unsqueeze(1),
                                                                                    dim=0).view(w_f, h_f, d_f)
                                            
                                elif args.prot_bank_data == 'all':
                                    if args.simce_loss and args.simce_data == 'sup':
                                        label_prot_bank = torch.mean(prototype_bank[:labeled_bs], dim=0)
                                        if not args.prot_un:
                                            label_prot_bank = F.normalize(label_prot_bank, dim=-1)

                                        mask_l = torch.zeros_like(y_all[i][labeled_bs:]).cuda()
                                        for ba in range(labeled_bs):
                                            for n_class in range(num_classes):
                                                class_prototype = label_prot_bank[n_class]
                                                mask_l[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                                        class_prototype.unsqueeze(1),
                                                                                        dim=0).view(w_f, h_f, d_f)
                                    if not args.prot_fusion:
                                        all_prot_bank = torch.mean(prototype_bank, dim=0)
                                        if not args.prot_un:
                                            all_prot_bank = F.normalize(all_prot_bank, dim=-1)
                                    else:
                                        label_prot_bank = torch.mean(prototype_bank[:labeled_bs], dim=0)
                                        unlabel_prot_bank = torch.mean(prototype_bank[labeled_bs:], dim=0)
                                        if not args.prot_un:
                                            all_prot_bank =  F.normalize(torch.cat([label_prot_bank.unsqueeze(0), unlabel_prot_bank.unsqueeze(0)], dim=0), dim=-1)
                                            all_prot_bank =  (all_prot_bank[0] + p_weight * all_prot_bank[1]) / (1 + p_weight) 
                                        else:
                                            all_prot_bank = (label_prot_bank + p_weight * unlabel_prot_bank) / (1 + p_weight)

                                    mask_t = torch.zeros_like(y_all[i]).cuda()
                                    for ba in range(batch_o):
                                        for n_class in range(num_classes):
                                            class_prototype = all_prot_bank[n_class]
                                            mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                                    class_prototype.unsqueeze(1),
                                                                                    dim=0).view(w_f, h_f, d_f)
                                            
                                elif args.prot_bank_data == 'sep':
                                    label_prot_bank = torch.mean(prototype_bank[:labeled_bs], dim=0)
                                    unlabel_prot_bank = torch.mean(prototype_bank[labeled_bs:], dim=0)
                                    if not args.prot_un:
                                        label_prot_bank = F.normalize(label_prot_bank, dim=-1)
                                        unlabel_prot_bank = F.normalize(unlabel_prot_bank, dim=-1)

                                    mask_t = torch.zeros_like(y_all[i]).cuda()
                                    for ba in range(labeled_bs):
                                        for n_class in range(num_classes):
                                            class_prototype = label_prot_bank[n_class]
                                            mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                                    class_prototype.unsqueeze(1),
                                                                                    dim=0).view(w_f, h_f, d_f)
                                    for ba in range(labeled_bs, batch_o):
                                        for n_class in range(num_classes):
                                            class_prototype = unlabel_prot_bank[n_class]
                                            mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                                    class_prototype.unsqueeze(1),
                                                                                    dim=0).view(w_f, h_f, d_f)
                                    mask_l = mask_t[:labeled_bs]
                                            
                                    
                                weight_pixel_t = (1 - mse_loss(mask_t, y_all[j])).mean(1)
                                weight_pixel_t = weight_pixel_t * mask

                                #plot
                                sim_index = mask_t.argmax(dim=1)

                                # prototype loss
                                if args.prot_cons:
                                    if args.prot_amb_mode == 'org':
                                        loss_t = consistency_criterion(y_all[i], torch.argmax(y_all[j], dim=1).detach())
                                        loss_consist += (loss_t * weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)

                                if args.prot_ual:
                                    if not args.prot_bank_alt:
                                        # unlabel_prot_bank align with label_prot_bank
                                        unlabel_prot_bank = unlabel_prot_bank.contiguous().view(c_o, -1)
                                        label_prot_bank = label_prot_bank.contiguous().view(c_o, -1)
                                        if args.prot_ual_det:
                                            label_prot_bank = label_prot_bank.detach()
                                        
                                        if not args.prot_ual_d:
                                            # positive
                                            ul_cos = 0
                                            for n_class in range(num_classes):
                                                ul_cosd = 1-F.cosine_similarity(unlabel_prot_bank[n_class], label_prot_bank[n_class], dim=0)
                                                # print(ul_cosd)
                                                ul_cos += ul_cosd

                                            u_sim_loss += ul_cos / num_classes

                                        else:
                                            pos_similarities = []
                                            neg_similarities = []

                                            # Calculate positive and negative similarities for each class
                                            for c1 in range(num_classes):
                                                # Positive similarity (unlabeled i-th vector with labeled i-th vector)
                                                pos_sim = F.cosine_similarity(unlabel_prot_bank[c1], label_prot_bank[c1], dim=0)
                                                pos_similarities.append(pos_sim.unsqueeze(0))
                                                
                                                # Loop through all other classes for negative pairs
                                                for c2 in range(num_classes):
                                                    if c1 != c2:
                                                        # Negative similarity (unlabeled i-th vector with labeled j-th vector)
                                                        neg_sim = F.cosine_similarity(unlabel_prot_bank[c1], label_prot_bank[c2], dim=0)
                                                        neg_similarities.append(neg_sim.unsqueeze(0))

                                            # Stack all positive and negative similarities
                                            pos_similarities = torch.cat(pos_similarities, dim=0).unsqueeze(1)
                                            neg_similarities = torch.cat(neg_similarities, dim=0).unsqueeze(1)

                                            # Combine positive and negative similarities into a similarity matrix
                                            sim_matrix = torch.cat([pos_similarities, neg_similarities], dim=1)

                                            # Create labels: 0 for positive pairs, 1 for negative pairs
                                            labels = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=sim_matrix.device)

                                            # Compute cross-entropy loss
                                            u_sim_loss += F.cross_entropy(sim_matrix, labels) / sim_matrix.shape[0]


            if args.mapping_loss:
                # answer: v_outputs, r_outputs, v_b_outputs, r_b_outputs
                # input : v_bd_seg_pred, r_bd_seg_pred, v_seg_bd_pred, r_seg_bd_pred

                if args.mapping_det:
                    seg_bd_pred_1 = seg_bd_pred_1.detach()
                    seg_bd_pred_2 = seg_bd_pred_2.detach()
                    bd_seg_pred_1 = bd_seg_pred_1.detach()
                    bd_seg_pred_2 = bd_seg_pred_2.detach()

                bd_seg_bd_pred1 = seg_bd_map(bd_seg_pred_1, input_type = 'seg', activation = True)['seg_bd_out']
                bd_seg_bd_pred2 = seg_bd_map(bd_seg_pred_2, input_type = 'seg', activation = True)['seg_bd_out']
                seg_bd_seg_pred1 = bd_seg_map(seg_bd_pred_1, input_type = 'bd', activation = False)['bd_seg_out']
                seg_bd_seg_pred2 = bd_seg_map(seg_bd_pred_2, input_type = 'bd', activation = False)['bd_seg_out']

                seg_bd_seg_pred1 = F.softmax(seg_bd_seg_pred1, dim=1)
                seg_bd_seg_pred2 = F.softmax(seg_bd_seg_pred2, dim=1)

                if args.mapping_ans_det:
                    # seg_ans1 = seg_o1.detach()
                    # seg_ans2 = seg_o2.detach()
                    seg_ans1 = y_all[0].detach()
                    seg_ans2 = y_all[1].detach()
                    bd_ans1 = F.tanh(bd_out1).detach()
                    bd_ans2 = F.tanh(bd_out2).detach()
                else:
                    # seg_ans1 = seg_o1
                    # seg_ans2 = seg_o2
                    seg_ans1 = y_all[0]
                    seg_ans2 = y_all[1]
                    bd_ans1 = F.tanh(bd_out1)
                    bd_ans2 = F.tanh(bd_out2)

                assert seg_ans1.shape == seg_bd_seg_pred1.shape
                assert bd_ans1.shape == bd_seg_bd_pred1.shape
                assert seg_ans2.shape == seg_bd_seg_pred2.shape
                assert bd_ans2.shape == bd_seg_bd_pred2.shape

                # boundary plot refinement
                v_b_tanR = torch.abs(bd_seg_bd_pred1)
                r_b_tanR = torch.abs(bd_seg_bd_pred2)

                
                bd_seg_map_loss = torch.mean(kl_loss(seg_bd_seg_pred1, seg_ans1)) + torch.mean(kl_loss(seg_bd_seg_pred2, seg_ans2))
                seg_bd_map_loss = torch.mean((bd_ans1 - bd_seg_bd_pred1))**2 + torch.mean((bd_ans2 - bd_seg_bd_pred2)**2)

                if (torch.any(torch.isnan(seg_bd_map_loss)) or torch.any(torch.isnan(bd_seg_map_loss)) ):
                    print('Loss nan find')
                    raise ValueError('Loss nan find')
                
                if args.mapping_ramp:
                    m_weight = get_current_consistency_weight(args.mapping_loss_weight, iter_num // 150)
                    writer.add_scalar('weight/mapping_weight', m_weight, iter_num)
                    loss += m_weight * (seg_bd_map_loss + bd_seg_map_loss*args.mapping_bd_seg_weight)
                else:
                    loss += args.mapping_loss_weight * (seg_bd_map_loss + bd_seg_map_loss*args.mapping_bd_seg_weight)

                writer.add_scalar('loss/seg_bd_map_loss', seg_bd_map_loss, iter_num)
                writer.add_scalar('loss/bd_seg_map_loss', bd_seg_map_loss, iter_num)


            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(args.consistency, iter_num // 150)
            writer.add_scalar('weight/consistency_weight', consistency_weight, iter_num)
            simce_weight = get_current_consistency_weight(args.sim_uramp_weight, iter_num // 150)
            writer.add_scalar('weight/sim_uramp_weight', simce_weight, iter_num)
            ual_weight = get_current_consistency_weight(args.prot_ual_weight, iter_num // 150)
            writer.add_scalar('weight/ual_weight', ual_weight, iter_num)
            # writer.add_scalar('weight/ual_cons', consistency_weight*ual_weight, iter_num)

            loss += consistency_weight * (loss_consist)
            if args.prot_ual:
                # loss += consistency_weight * u_sim_loss * ual_weight
                loss +=  u_sim_loss * ual_weight
            if args.simce_loss: 
                loss += loss_cs* args.simce_weight
                if args.simce_data == 'all':
                    if args.sim_uramp:
                        loss += uloss_cs* simce_weight
                    else:
                        loss +=  uloss_cs* args.simce_weight


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            if args.prot_ual:
                writer.add_scalar('Co_loss/u_sim_loss', u_sim_loss, iter_num)
            if args.simce_loss:
                writer.add_scalar('simce_loss/loss_cs', loss_cs, iter_num)
                if args.simce_data == 'all':
                    writer.add_scalar('simce_loss/uloss_cs', uloss_cs, iter_num)
            # if args.prot_sim:
            #     writer.add_scalar('Co_loss/d_loss', d_loss, iter_num)

            if iter_num % 100 == 0 and args.plot:
                for item in grid:
                    writer.add_image('train/'+item, grid[item], iter_num)

            ######## validation  #################
            if args.dataset_name == 'LA':
                list_name = '/data/LA/test.list'   # todo change test flod
                with open(args.root_path + list_name, 'r') as f:                                         # todo change test flod
                    test_image_list = f.readlines()
                test_image_list = [args.root_path +'/LA_data/' + item.replace('\n', '')+"/mri_norm2.h5" for item in test_image_list] # todo change test flod
            
            elif args.dataset_name == 'Pancreas_CT':
                list_name = '/data/Pancreas/test.list'   # todo change test flod
                with open(args.root_path + list_name, 'r') as f:                                         # todo change test flod
                    test_image_list = f.readlines()
                test_image_list = [args.root_path +'/Pancreas_h5/' + item.replace('\n', '') + '_norm.h5' for item in test_image_list]         # todo change test flod

            if iter_num >= 400 and iter_num % 400 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model,
                                                          test_list = test_image_list,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=18,
                                                          stride_z=4,
                                                          dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model,
                                                          test_list = test_image_list,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    # torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                    early_stop = 0
                else:
                    early_stop += 1
                    print('Current Early stop:{}, iteration: {}'.format(early_stop, iter_num))
                    
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break


            ## change lr
            if args.clr:
                if early_stop % 15 == 0 and early_stop != 0:
                # if iter_num % 2500 == 0 and iter_num!= 0:
                    lr_ = lr_ * 0.9
                    print('lr change to:', lr_)
                    lr_ = max(lr_, lr_ * (0.1**3))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_


        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", args =  args)
            iterator.close()
            break
        
    
    time2 = time.time()
    logging.info('training time: %f' % (time2 - time1))
    writer.close()