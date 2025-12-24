import h5py
import math
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib
import argparse



def get_task_name(args):

    # task_name += f'_{args.dataset}'

    task_name = f'PMCM_{args.dataset_name}_{args.labelnum}_{args.max_iteration}'

    task_name += f'_{args.lamda}'


    if args.l_loss != 'org':
        task_name += f'_{args.l_loss}'

    if args.nramp:
        task_name += f'_nr'
    

    if args.bd_loss:
        task_name += f'_bdloss-w{args.bd_loss_weight}_ch{args.bd_channel}'
        if args.bd_entropy_weighted:
            task_name += f'_entw'
            if args.bd_entropy_weighted_one:
                task_name += f'_one'

        if args.task_cons:
            task_name += f'_task_w{args.task_cons_weight}_{args.task_bd_seg_weight}'
            if args.task_net_pred:
                task_name += f'_netpred_{args.task_net_mode}_{args.task_update}'
                if args.task_update == 'unc':
                    task_name += f'_{args.task_unc_pse}'
                task_name += f'_{args.task_criteria}'
                if args.task_net_siamese_map:
                    task_name += f'_siam_{args.mapping_dim}'
                if args.task_net_map_pred:
                    task_name += f'_mappred'
                if args.task_net_detach:
                    task_name += f'_det'
                if args.task_prot:
                    task_name += f'_prot'
                if args.task_ent:
                    task_name += f'_ent_{args.task_ent_mode}'
                    if args.task_ent_mode == 'unc':
                        task_name += f'_{args.task_unc_pse}'
                    if args.task_ent_det:
                        task_name += f'_det'
                if args.task_c:
                    task_name += f'_c'


                if args.task_lca:
                    task_name += f'_lca_{args.task_ag_ent_mode}'
                    if args.task_ag_erl:
                        task_name += f'_erl'
                        if args.task_ag_erl_mode != 'org':
                            task_name += f'_{args.task_ag_erl_mode}'
                    if args.task_ag_ent_det:
                        task_name += f'_det'
                if args.task_ag:
                    task_name += f'_ag_{args.task_ag_weight}_{args.task_ag_criteria}_{args.task_ag_ans}_{args.task_ag_mode}_{args.task_ag_pse}'
                    task_name += f'_{args.task_ag_seg_weight}_{args.task_ag_bd_weight}'
                    if args.task_ag_sharp:
                        task_name += f'_sharp'
                    if args.task_ag_ent:
                        task_name += f'_ent_{args.task_ag_ent_mode}'
                        if args.task_ag_erl:
                            task_name += f'_erl'
                            if args.task_ag_erl_mode != 'org':
                                task_name += f'_{args.task_ag_erl_mode}'
                        if args.task_ag_entmin:
                            task_name += f'_min_{args.task_ag_entmin_mode}'
                        if args.task_ag_ent_det:
                            task_name += f'_det'

            else:
                task_name += f'_ch{args.bd_channel}_{args.task_cons_data}_{args.task_update}'
                task_name += f'_{args.task_cons_mode}'

                if args.task_detach:
                    task_name += f'_det'

                if args.task_entropy_weighted: 
                    task_name += f'_entw'
                    if args.task_entropy_weighted_one:
                        task_name += f'_one'
                    task_name += f'_{args.task_ent_mode}'

                if args.bd_function == 'sigmoid':
                    task_name += f'_sig'
                else:
                    task_name += f'_soft'
                
        
    if args.task_exp:
        task_name += f'_exp_{args.task_exp_type}'

    if args.prot:            
        task_name += f'_prot'
        if args.prot_cons:
            task_name += f'_cons_{args.consistency}_{args.prot_mode}'

        task_name += f'_{args.prot_data}'
        task_name += f'_{args.prot_bank_data}'

        if args.prot_fusion:
            task_name += f'_fu'

        if args.prot_un:
            task_name += f'_un'
        if args.prot_ual:
            task_name += f'_ual_{args.prot_ual_weight}'
            if args.prot_ual_d:
                task_name += f'_d'

        if args.prot_amb_mode != 'org':
            task_name += f'_amb_{args.prot_amb_mode}'


    if args.feat_sim:
        task_name += f'_fs_{args.feat_sim_weight}'

    if args.mapping_loss:
        task_name += f'_mloss_{args.mapping_loss_weight}_{args.mapping_bd_seg_weight}_{args.mapping_loss_type}'
        if args.mapping_ramp:
            task_name += f'_ramp'
        if args.mapping_det:
            task_name += f'_det'
        if args.mapping_ans_det:
            task_name += f'_ansdet'
    
    # if args.mapping_id:
    #     task_name += f'_id_{args.mapping_id_weight}'
        
                
    return task_name