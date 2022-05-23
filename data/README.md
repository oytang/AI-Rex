# data of chemical reactions

- `cleaned_uspto50k.csv` 包括了我们目前需要的所有真反应（正例）
- `negative_strict.csv` 第一类负例 通过将一个反应的反应模板在反应物上进行错位匹配生成 代表了立体/区位选择性错误的反应
- `negative_random.csv` 第二类负例 通过将一个反应与其他反应的模板进行匹配生成 代表了反应物发生反应的更多可能 但实际的产物是动力学/热力学最优的产物 可以认为其它产物不足以生成 生成时对一个反应随机筛选了一定数量的模板与之进行匹配
- `negative_random_comprehensive.csv` 可通过compiler来进行整合生成 使用方法如下 该文件也是包含第二类负例 唯一不同在于对于每一个反应遍历了所有USPTO-50k数据集中的模板 因此是该数据集基础上可以生成的最完备的负例数据集
    ```
    python compile_random_comprehensive.py
    ```
- `ECFP-split.py` 可以根据反应物的分子指纹ECFP来将反应进行聚类然后进行train-valid-test-split 同时也做了更多一些预处理来最大限度上避免数据泄露 使用方法如下
    ```
    python ECFP-split.py [--help] [--num_false 42000 #从整个随机负例中采样的数量] [--train_size 0.7] [--valid_size 0.2] [--test_size 0.1] [--ECFP4nBits 1024 #分子指纹二元向量的长度]
    ```