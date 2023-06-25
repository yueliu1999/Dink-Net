


<div align="center">

<h2><a href="https://arxiv.org/pdf/2305.18405.pdf">Dink-Net: Neural Clustering on Large Graphs</a></h2>

[Yue Liu](https://yueliu1999.github.io/)<sup>1,2</sup>, [Ke Liang](https://liangke23.github.io/)<sup>1</sup>,  [Jun Xia](https://junxia97.github.io/)<sup>2</sup>, [Sihang Zhou](https://scholar.google.com/citations?user=p9Se8kYAAAAJ&hl=zh-CN&oi=ao/)<sup>1</sup>, [Xihong Yang](https://xihongyang1999.github.io/)<sup>1</sup>, [Xinwang Liu](https://xinwangliu.github.io/)<sup>1</sup>, [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>

<sup>1</sup>[National University of Defense Technology](https://english.nudt.edu.cn/), <sup>2</sup>[Westlake University](https://westlake.edu.cn/)


</div>




<p align="center"> 
<a href="https://pytorch.org/" alt="PyTorch">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /> 
</a>
<a href="https://icml.cc/Conferences/2023" alt="Conference">
<img src="https://img.shields.io/badge/ICML'23-brightgreen" />
</a>
[![stars](https://img.shields.io/github/stars/yueliu1999/Dink-Net?color=yellow)](https://github.com/yueliu1999/Dink-Net/stars)
[![forks](https://img.shields.io/github/forks/yueliu1999/Dink-Net?color=lightblue)](https://github.com/yueliu1999/Dink-Net/forks)
[![ issues](https://img.shields.io/github/issues-raw/yueliu1999/Dink-Net?color=%23FF9600)](https://github.com/yueliu1999/Dink-Net/issues)
[![ visitors](https://visitor-badge.glitch.me/badge?page_id=yueliu1999.Dink-Net)](https://github.com/yueliu1999/Dink-Net)
</p>


Deep graph clustering, which aims to group the nodes of a graph into disjoint clusters with deep neural networks, has achieved promising progress in recent years. However, the existing methods fail to scale to the large graph with million nodes. To solve this problem, a scalable deep graph clustering method (*Dink-Net*) is proposed with the idea of <u>di</u>lation and shri<u>nk</u>. Firstly, by discriminating nodes, whether being corrupted by augmentations, representations are learned in a self-supervised manner. Meanwhile, the cluster centers are initialized as learnable neural parameters. Subsequently, the clustering distribution is optimized by minimizing the proposed cluster dilation loss and cluster shrink loss in an adversarial manner. By these settings, we unify the two-step clustering, i.e., representation learning and clustering optimization, into an end-to-end framework, guiding the network to learn clustering-friendly features. Besides, *Dink-Net* scales well to large graphs since the designed loss functions adopt the mini-batch data to optimize the clustering distribution even without performance drops. Both experimental results and theoretical analyses demonstrate the superiority of our method.









## Citation

If you find this repository helpful, please cite our papers.

```
@inproceedings{Dink-Net,
  title={Dink-Net: Neural Clustering on Large Graphs},
  author={Liu, Yue and Liang, Ke and Xia, Jun and Zhou, Sihang and Yang, Xihong and Liu, Xinwang and Li, Stan Z.},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```
