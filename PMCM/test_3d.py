import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch import test_all_case


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')

parser.add_argument('--exp', type=str, default='MCNet', help='exp_name')
parser.add_argument('--model', type=str, default='mine3d_v1', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=0, help='print metrics for every samples?')
# parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

parser.add_argument('--root_path', type=str, default='/media/bspubuntu/1TBSSD/A_exp/dataset', help='Name of Experiment')               # todo change dataset path
parser.add_argument('--dataset_name', type=str, choices=["LA", "Pancreas_CT"], default='Pancreas_CT', help='dataset')               # todo change dataset path
parser.add_argument('--cv', type=str,  default='0', help='Cross Validation')
parser.add_argument('--labelnum', type=int,  default=6, help='trained samples')
parser.add_argument('--n_class', type=int,  default=2)
parser.add_argument('--mode', type=str, choices=["best", "iter"], default='best') 
parser.add_argument('--model_name', type = str, default='MR_Pancreas_CT_6_bdloss-w1.0_ch1', help='model name')
parser.add_argument('--bd_loss', action = 'store_true',  default=False, help='boundary Loss')
parser.add_argument('--simsiam', action='store_true', default=False)

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def get_task_name(args):

    # task_name += f'_{args.dataset}'
    task_name = f'../{args.model_name}'
    task_name += f'/{args.mode}'
    task_name += '.csv'

    return task_name

if FLAGS.cv == '0':
    print('******************************* Start Exp Inf *******************************  ')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../" + FLAGS.model_name + '/MR_flod' + str(FLAGS.cv) + '/'  # todo change test flod
test_save_path = "../" + FLAGS.model_name +"/prediction/"  # todo change test flod
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)



# FLAGS.exp = f'mRcnet3d_v1_flod_cv{FLAGS.cv}_{FLAGS.dataset_name}_{FLAGS.labelnum}' 
# snapshot_path = "../" + FLAGS.exp + "/"
# test_save_path = f"../{FLAGS.exp}/predictions/"




num_classes = 2
if FLAGS.dataset_name == 'LA':
    patch_size = (112, 112, 80)
    list_name = '/LA/Flods/test' + str(FLAGS.cv) + '.list'   # todo change test flod
    with open(FLAGS.root_path + list_name, 'r') as f:                                         # todo change test flod
        image_list = f.readlines()
    image_list = [FLAGS.root_path +'/LA_data/' + item.replace('\n', '')+"/mri_norm2.h5" for item in image_list] # todo change test flod
elif FLAGS.dataset_name == 'Pancreas_CT':
    patch_size = (96, 96, 96)
    list_name = '/Pancreas/Flods/test' + str(FLAGS.cv) + '.list'   # todo change test flod
    with open(FLAGS.root_path + list_name, 'r') as f:                                         # todo change test flod
        image_list = f.readlines()
    image_list = [FLAGS.root_path +'/pancreas_data/' + item for item in image_list]         # todo change test flod


# if FLAGS.dataset_name == "LA":
    
#     FLAGS.root_path = FLAGS.root_path + 'data/LA'
#     with open(FLAGS.root_path + '/test.list', 'r') as f:
#         image_list = f.readlines()
#     image_list = [
#         FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list
#     ]

# elif FLAGS.dataset_name == "Pancreas_CT":
    
#     FLAGS.root_path = FLAGS.root_path + 'data/Pancreas/'
#     # FLAGS.root_path = '/home/jwsu/semi' + 'data/Pancreas'
#     with open(FLAGS.root_path + '/test.list', 'r') as f:
#         image_list = f.readlines()
#     image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)


def test_calculate_metric():

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test", args=FLAGS)
    # net.cuda()
    if FLAGS.mode == 'best':
        save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    else:
        save_mode_path = os.path.join(snapshot_path, 'iter_6000.pth'.format(FLAGS.model))
    # save_mode_path = '/mnt/imtStu/jwsu/Rectify/LA_rectify_prototype_4_labeled_topk_mine3d_v1/1000/mine3d_v1_best_model.pth'
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model,
                                   1,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(112, 112, 80),
                                   stride_xy=18,
                                   stride_z=4,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms,
                                   dataset_name = FLAGS.dataset_name)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=True,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms,
                                   dataset_name = FLAGS.dataset_name)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

    import pandas as pd
    
    # csv_path = f"../model/MR_{FLAGS.dataset_name}.csv"
    # csv_path = f"../{FLAGS.dataset_name}_{FLAGS.labelnum}_{FLAGS.mode}.csv"
    # csv_path = +f"MR_{FLAGS.mode}.csv"

    # csv_path =  get_task_name(FLAGS)

    csv_path =  get_task_name(FLAGS)


    check_file = os.path.isfile(csv_path)
    if check_file:
        df = pd.read_csv(csv_path)

    result_df = pd.DataFrame(columns=['CV', 'Dice', 'Jaccard', 'HD', 'ASD' ])
    result_df.CV = pd.Series(int(FLAGS.cv))
    result_df.Dice = pd.Series(metric[0])
    result_df.Jaccard = pd.Series(metric[1])
    result_df.HD = pd.Series(metric[2])
    result_df.ASD = pd.Series(metric[3])

    if not check_file:
        result_df.to_csv(csv_path, index=False)
    else:
        frames = [df, result_df]
        result_df = pd.concat(frames)
        

        if FLAGS.dataset_name == 'LA':    #todo change
            t_fold = 4
        else:
            t_fold = 3

        if int(FLAGS.cv) == t_fold:
            result_df['Dice_avg_sd'] = str((result_df["Dice"].mean()*100).round(2)) + '±' + str((result_df["Dice"].std()*100).round(2))
            # result_df['Dice_std'] = (result_df["Dice"].std()*100).round(2)

            result_df['Jaccard_avg_sd'] = str((result_df["Jaccard"].mean()*100).round(2)) + '±' + str((result_df["Jaccard"].std()*100).round(2))
            # result_df['Jaccard_std'] = (result_df["Jaccard"].std()*100).round(2)

            result_df['HD_avg_sd'] = str(result_df["HD"].mean().round(2)) + '±' + str(result_df["HD"].std().round(2))
            # result_df['HD_std'] = result_df["HD"].std()

            result_df['ASD_avg_sd'] = str(result_df["ASD"].mean().round(2)) + '±' + str(result_df["ASD"].std().round(2))
            # result_df['ASD_std'] = result_df["ASD"].std()


        result_df.to_csv(csv_path, index=False)
