# Baseline Transformer

模型文件放在百度网盘：
链接:https://pan.baidu.com/s/17cvlGIv64GaSUh1KHqcbSA  密码:l8l6

pt文件是使用MIT-USPTO数据集训练的模型

测试时采用Molecular Transformer原Github中的设置，将model参数改为模型文件所在路径即可

# 0521新增用于分类的Transformer模型

Molecular transformer源代码网址：
https://github.com/pschwllr/MolecularTransformer

模型pt文件放在百度网盘：
链接:https://pan.baidu.com/s/12FAlRdYy1udAigVb5GZjFA  密码:3mjx

运行时，将ipynb文件放在Molecular Transformer文件夹下，All_Data.zip解压在./data文件夹下，pt文件放在./experiments/checkpoints/All_Data_binary下。

若从头训练模型，直接逐步运行ipynb即可，若使用预训练模型，确认模型和数据的路径正确后，从make predictions部分开始运行即可。
