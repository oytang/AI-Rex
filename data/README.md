# data of chemical reactions

- `cleaned_uspto50k.csv` 包括了我们目前需要的所有真反应（正例）
- `negative_strict.csv` 第一类负例 通过将一个反应的反应模板在反应物上进行错位匹配生成 代表了立体/区位选择性错误的反应
- `negative_random.csv` 第二类负例 通过将一个反应与其他反应的模板进行匹配生成 代表了反应物发生反应的更多可能 但实际的产物是动力学/热力学最优的产物 可以认为其它产物不足以生成
