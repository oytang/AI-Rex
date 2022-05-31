[![backgroud](https://img.shields.io/badge/ai4science-Chemistry-blue)](https://ai4science.io/chemistry_zh.html)
[![branch](https://img.shields.io/badge/Branch-main-green)](https://github.com/Chemino/AI-Rex)
[![branch](https://img.shields.io/badge/Branch-dev-yellow)](https://github.com/Chemino/AI-Rex/tree/dev)

<a href="url"><img src="./misc/AIRex-logos.jpeg" align="left" height="150" width="150" ></a>

# AI.Rex

An integrated framework for chemical reaction feasibility prediction. This is a structured solution for one of the Chemistry Challenges of [ai4science hackathon](https://ai4science.io/), organized by [DeepVerse](deepverse.tech/en/), [Bota Bio](www.bota.bio), [Chemical.AI](https://chemical.ai/) and [Foreseen Biotechnology](www.foreseepharma.com/en-us).

## Team members
[@Chemino](https://github.com/Chemino)
[@shenwanxiang](https://github.com/shenwanxiang)
[@Mercury1874](https://github.com/1874Mercury)
[@jiayujack](https://github.com/jiayujack)
[@LHZ-HUST](https://github.com/LHZ-HUST)
[@CreamyLong](https://github.com/CreamyLong)


## Challenge description

Evaluate reactivity/reaction feasibility with quantum mechanics and/or AI technology.

- Predict whether the reaction is feasible based on public reaction data such as Chemical reactions from US patents
- Collect or generate failure reactions to train and/or evaluate the model
- It is encouraged to use quantum mechanics approaches.

## Alternative Git repository

- Without full commit history and development-relevant content
- https://github.com/Chemino/AI.Rex

## Highlights

In chemical synthesis planning, retrosynthsis analysis tools can readily provide numerous synthetic paths, but that is beyond the capacity of experimental validation. Hence, evaluating the reaction feasibility accuractely and efficiently is vital in reducing the number of proposed paths and only retaining those have higher chance of success in wet lab.
In this work, we presented several novel ideas, with the hope that they can inspire future work:
- We designed neural network models based on Reaction SMILES sequence data or graph data, which can represent molecular structures in reactions with higher fidelity than molecular fingerprints such as ECFP;
- We systemically generated negative samples (unfeasible reactions) in form of Reaction SMILES, by mapping correct reaction template to incorrect reaction site, or by mapping incorrect reaction template, which is the type of unfeasible reactions one can make based on a given dataset that can cause larest confusion;
- We suggested ECFP-based train/valid/test split, which can make the prediction task even harder and can potentially enhance the generalizability of the models;
- We estabilished an sequence-based feasibility prediction model leveraging attention mechanism (built upon [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer)), reaching a high test accuracy as \~98%;
- We developed [RRD package](https://github.com/Chemino/AI-Rex/tree/main/RRD), that can be used to calculate Reactivity-Related bond/atom-wise Descriptors with graph-based models; It can be easily integrated into any other molecular modeling pipeline in a "plug and play" manner, and can compute Quantum Mechanics descriptors on-the-fly.
- We estabilished a graph-based convolutional feasibility prediction model (built upon [TAGCN](https://arxiv.org/pdf/1710.10370.pdf)), reaching a reasonably high test accuracy \~91%; After adding in atom/bond features computed by RRD, the test accuracy increased by \~1.0%.

在化学合成路线规划中，逆合成分析工具可以很容易地提供许多合成路径，但这超出了实验验证的可行范围。因此，准确有效地评估反应的可行性对于减少建议的路径数量并只保留那些在湿式实验室中具有较高成功机会的路径至关重要。在这项工作中，我们提出了几个新的想法，希望它们能对未来的工作有所启发。

- 设计了基于反应SMILES序列数据或图数据的神经网络模型，它可以比ECFP等分子指纹更保真地表示反应中的分子结构；
- 以Reaction SMILES的形式，通过将正确的反应模板映射到不正确的反应位点，或通过映射不正确的反应模板，系统地生成负面样本（即不可行的反应），这是基于给定数据集所能做出的最具混淆性的不可行反应类型；
- 提出了基于ECFP的训练/验证/测试数据集分割，这可以使预测任务更加困难，并可以潜在地提高模型的泛化能力；
- 建立了一个基于序列的可行性预测模型，该模型利用了注意力机制（建立在[Molecular Transformer](https://github.com/pschwllr/MolecularTransformer)的基础上），达到了很高的测试准确率，约为98%；
- 开发了[RRD代码包](https://github.com/Chemino/AI-Rex/tree/main/RRD)，可用于计算基于图的模型的活性相关的键/原子描述符；它可以很容易地以“即插即用”的方式集成到任何其他分子建模管道中，并可以即时计算量子力学描述符；
- 建立了一个基于图卷积的可行性预测模型（建立在[TAGCN](https://arxiv.org/pdf/1710.10370.pdf)的基础上），达到了相当高的测试精度\~91%；在加入由RRD计算的原子/键特征后，测试精度增加了\~1.0%。

## 详细方法

### 负例生成

由于过往文献较少报道不可行反应，我们需要从可行反应中提取隐含的信息，即可行反应的未记载的可能产物即视为反应不发生，因为它们未能竞争过主要产物。因此，我们对数据集内每一条反应的反应物应用了大量的反应模板以寻找可能产物，并将非数据集记载产物的产物视为负例。该过程使用了rdkit与rdchiral包处理化学信息。

### 基于ECFP的数据集划分

分子指纹ECFP提供了一种将SMILES转化为二元向量的一种策略，这样我们就可以测量不同反应的反应物之间的相似度。为了尽可能避免数据泄露，我们采取了基于ECFP的数据集划分策略：先将所有反应通过反应物的ECFP4指纹进行聚类，然后将反应物相似的反应同时划分到训练/验证/测试任一数据集内，以增加预测难度，增强模型的泛化能力
。
### 基于transformer的序列算法预测反应可行性

在USPTO-MIT数据集上开发了一种用于预测反应生成物的生成式模型和一种用于预测反应能否发生的判别式模型。两种模型均是基于机器翻译包onmt，由4层编码器结构和4层解码器结构组成。生成式模型把预测化学反应产物建模为从反应物到生成物的翻译问题，判别式模型把预测反应能否发生建模为输出反应成功概率的回归问题.

### 基于图神经模型预测反应可行性

反应性的可行性很大程度上决定于反应的热力学和动力学壁垒。因此，原子和键的热力学和动力学特征将对反应的可行性起到决定性的作用。普通的GNN网络用于表征小分子忽略了反应性，因为表征分子只用了基本的原子与键的属性，比如原子类型、键类型等。

### 量子化学描述符信息的嵌入

在我们的模型中，我们首先使用模型、或者文献中相关公式，计算了与反应相关的原子与键的描述符（RRD）。由于基于DFT的计算速度很慢，我们集成了基于GNN的原子、键属性预测模型， 来on-the-fly快速得到与反应相关的四大关键描述符（空间位阻、静电势、福井指数、键能）， 用于进一步的反应预测。

在反应预测的模型上，我们使用两个分子图分别表示的反应物和产物，图中的每个节点代表原子，边表示化学键。化学反应可行性预测被建模为一个用图神经网络预测的二分类任务。原子的特征包含我们上述提出的RRD中的量化特征。在图神经网络部分，我们选择了TAGCN和Graph transformer两种图模型分别模型搭建。反应物图和产物图先输入图神经网络学习到各自的图表示，最后用几层MLP模拟反应物和产物之间的交互作用，最后输出预测结果。

## File Structure

    .
    ├── data                           # dataset, negative data generation, data split
    ├── misc                           # meeting history, relevant papers, data exploration  
    ├── RRD                            # a package to compute the Reactivity-Related bond/atom-wise Descriptors (RRD)
    ├── TAGConv                        # a graph-based model, with baseline and QM-embedded implementation
    ├── Transformer                    # a squence-based model, built upon Molecular Transformer
    ├── LICENSE
    └── README.md
