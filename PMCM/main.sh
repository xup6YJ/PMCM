
python train.py --dataset_name LA --model mine3d_v1 --exp mine --labelnum 8 --gpu 0 --batch_size 4 --labeled_bs 2 --max_iteration 15000 \
    --lamda 1 \
    --task_cons_weight 0.05 --task_bd_seg_weight 1 --bd_loss --task_cons  --task_update own  --task_criteria kl \
    --consistency 1.5 --prot --prot_cons --prot_mode org  --prot_data all  --prot_bank_data all --prot_amb_mode org --prot_fusion  \
    --prot_ual_weight 0.7 --prot_ual  --prot_ual_d \
    --feat_sim_weight 0.1 --feat_sim  \
    --mapping_loss_weight 0.005 --mapping_loss  --mapping_ans_det --mapping_loss_type kl  --mapping_ramp \
    # --plot

python test_3d2.py --dataset_name LA  --model mine3d_v1 --exp mine --labelnum 8 --gpu 0 --mode best  \
    --model_name PMCM_LA_8_15000
# MR_LA_8_15000_1.0_bdloss-w1.0_ch1_task_w0.05_1.0_netpred_ind_own_kl_siam_8_det_prot_cons_1.5_org_all_all_fu_ual_0.7_d_fs_0.1_mloss_0.005_1_kl_ramp_ansdet



python train.py --dataset_name Pancreas_CT --model mine3d_v1 --exp mine --labelnum 6 --gpu 0 --batch_size 4 --labeled_bs 2 --max_iteration 15000 \
    --lamda 0.5 \
    --task_cons_weight 0.01 --bd_loss --task_cons   \
    --task_exp  --task_exp_type dtc  \
    --consistency 2 --prot --prot_cons --prot_mode org  --prot_data all  --prot_bank_data all --prot_amb_mode org --prot_fusion  \
    --prot_ual --prot_ual_weight 0.3 --prot_ual_d \
    --feat_sim --feat_sim_weight 0.01 \
    --mapping_loss_weight 0.005 --mapping_loss  --mapping_ans_det --mapping_loss_type kl  --mapping_ramp \

python test_3d2.py --dataset_name Pancreas_CT  --model mine3d_v1 --exp mine --labelnum 6 --gpu 0 --mode best  \
    --model_name PMCM_Pancreas_CT_6_15000
# MR_Pancreas_CT_6_15000_0.5_bdloss-w1.0_ch1_task_w0.01_1.0_netpred_ind_own_kl_siam_8_det_prot_cons_2.0_org_all_all_fu_ual_0.3_d_fs_0.01_mloss_0.005_1_kl_ramp_ansdet







