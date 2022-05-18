# ai4science-chemistry

## 重要资源

- [IBM MolecularTransformer](https://github.com/pschwllr/MolecularTransformer) 根据反应物预测产物、置信度
- [区位选择性预测](https://github.com/yanfeiguan/reactivity_predictions_substitution)
- [qmdesc](https://pypi.org/project/qmdesc/) 用以计算量子化学描述符 可以直接pip
- [USPTO-MIT](https://github.com/wengong-jin/nips17-rexgen) 在Lowe 2012的原版USPTO数据集上经过清洗得到的有效性更强的数据集

### [0516 导师交流会议录像链接 (44M)](https://uchicagoedu-my.sharepoint.com/:v:/g/personal/yifengt_uchicago_edu/EWXOFShl9sdMssk4ZDoS5yoBiktqLY_4DRT-clx-qnn1UA?e=xtajvT)

注：通过录像提取的音频已经push在git的根目录 `0516_chemistry_mentor_meeting.mp3` 所以除非需要看某个时候讲话人具体是谁 否则音频就够用了

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

## 资源

[filter_policy](https://figshare.com/articles/dataset/A_quick_policy_to_filter_reactions_based_on_feasibility_in_AI-guided_retrosynthetic_planning/13280507) 这里有几个已经经过训练的filter policy网络（也即反应可行性判断）我们可以用来benchmarking
