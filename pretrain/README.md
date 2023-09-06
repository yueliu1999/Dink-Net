
### Pretraining

Pretrain Dink-Net on your own data.


run codes with comments

```
python pretrain.py --device cuda:0 --dataset cora --hid_units 512 --lr 1e-3 --epochs 200

python pretrain.py --device cuda:0 --dataset citeseer --hid_units 1536 --lr 5e-4 --epochs 200

python pretrain.py --device cuda:0 --dataset amazon_photo --hid_units 512 --lr 1e-3 --epochs 100

python pretrain.py --device cuda:0 --dataset ogbn_arxiv --hid_units 1500 --encoder_layer 3 --lr 1e-4 --epochs 1 --batch_train --batch_size 8192 --eval_inter 1
```
copy the pretrained model parameters DinkNet_xxx.pt to ../models
