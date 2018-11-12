`
python eval_dir.py --img_dir ~/dataset/processed/ahe --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 1
`

`python eval_dir_refine.py --img_dir ~/dataset/processed/ahe --anno ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/all_cls_anno.json --topk 1`

`python libs/finetune.py --gpu 0 ~/dataset/Exclusively-Dark-Image-Dataset/ExDark/splits`