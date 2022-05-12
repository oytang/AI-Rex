# ai4science-chemistry

## 挑战题 #3
使用AI 技术预测反应可行性。
- 可以使用些公开的反应数据作为训练集，如美国专利数据。[Chemical reactions from US patents (1976-Sep2016)](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873/1)
  - 其中CML文件与SMILES文件是同样内容不同格式，我们只需要SMILES即可
  - grant与application包含的反应有较多重复，我认为[grant](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873/1?file=8664379)包含的反应更可靠一些
  - 仅这一个文件提供的反应数量足够大，一共1,808,937条目，暂不用再去找别的数据集
  - 由于该文件内很多反应重复或非常相似，可能需要批量清洗
  - `USPTO_data_structure.ipynb`可以看到数据集的一个overview
- 算法能够生成一些负例用于模型的训练或者验证。
- 可以融合量子化学的手段，比如在AI模型中结合量化描述符（如静电势、福井函数等）来辅助预测等多类形式。
