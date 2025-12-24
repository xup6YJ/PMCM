import torch
from torch import nn
import torch.nn.functional as F


class MappingNet(nn.Module):
    def __init__(self, args):
        super(MappingNet, self).__init__()

        n_classes = args.n_class
        self.mode = args.task_net_mode

        if self.mode == 'dep':
            self.map = nn.Sequential(
                nn.Conv3d(n_classes, 8, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0)
                )

        else:
            self.map_seg2bd = nn.Sequential(
                nn.Conv3d(n_classes, 8, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0)
                )
            
            self.map_bd2seg = nn.Sequential(
                nn.Conv3d(n_classes, 8, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0)
                )


        self.__init_weight()


    def forward(self, seg_input, bd_input):
        output = {}
        seg_input = F.softmax(seg_input, dim=1)
        bd_input = F.tanh(bd_input)

        if self.mode == 'dep':
            seg_pred_bd = self.map(seg_input)
            output['seg_bd_out'] = F.tanh(seg_pred_bd)
            
            bd_pred_seg = self.map(bd_input)
            output['bd_seg_out'] = F.softmax(bd_pred_seg, dim=1)

        else:
            seg_pred_bd = self.map_seg2bd(seg_input)
            output['seg_bd_out'] = F.tanh(seg_pred_bd)

            bd_pred_seg = self.map_bd2seg(bd_input)
            output['bd_seg_out'] = F.softmax(bd_pred_seg, dim=1)


        return output

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





class MappingNet_single_task(nn.Module):
    def __init__(self, args):
        super(MappingNet_single_task, self).__init__()

        n_classes = args.n_class
        self.mode = args.task_net_mode
        self.mapping_dim = args.mapping_dim

        self.map = nn.Sequential(
            nn.Conv3d(n_classes, self.mapping_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(self.mapping_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.mapping_dim, n_classes, kernel_size=1, stride=1, padding=0)
            )


        self.__init_weight()


    def forward(self, input, input_type = None, activation = True, id = False):
        output = {}
        if activation:
            if input_type == 'seg':
                input = F.softmax(input, dim=1)
            else:
                input = F.tanh(input)

        pred = self.map(input)

        if not id:
            if input_type == 'seg':
                output['seg_bd_out'] = F.tanh(pred)
            else:
                output['bd_seg_out'] = pred
        else:
            if input_type == 'seg':
                output['seg_seg_out'] = F.softmax(pred, dim=1)
            else:
                output['bd_bd_out'] = F.tanh(pred)


        return output

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m,nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()