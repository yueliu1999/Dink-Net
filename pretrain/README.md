
### Pretraining

Pretrain Dink-Net on your own data.


run codes with comments

```
python pretrain.py --device cuda:0 --dataset cora --hid_units 512 --lr 1e-3 --epochs 200 --seed 47 --km_cuda

python pretrain.py --device cuda:0 --dataset citeseer --hid_units 1536 --lr 5e-4 --epochs 200 --km_cuda

python pretrain.py --device cuda:0 --dataset amazon_photo --hid_units 512 --lr 1e-3 --epochs 500 --km_cuda

python pretrain.py --device cuda:0 --dataset ogbn_arxiv --hid_units 1500 --encoder_layer 3 --lr 1e-4 --epochs 1 --km_cuda
```
copy the pretrained model parameters DinkNet_xxx.pt to ../models
