`
python eval_dir.py --img_dir ~/dataset/processed/ahe --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 1
`

`python eval_dir_refine.py --img_dir ~/dataset/processed/ahe --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 1`

`python libs/finetune.py --gpu 0 ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/splits`

`python libs/finetune.py --gpu 0 /home/xinyu/dataset/yifan/enhanced_Exdark`

`python eval_dir_finetune.py --img_dir ~/dataset/processed/edark/single_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1/test_200/images/ --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 5`

`python libs/eval_dir_finetune.py --img_dir ~/dataset/processed/edark/single_unet_conv_add_vary_Tresidual_attention_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1/test_200/images/ --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 5 --resume_file finetuned_resnet50/model_best.pth.tar`