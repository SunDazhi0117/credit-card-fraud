# Project Introduction

## English: approximately 90 seconds

I built a credit card fraud detection project to explore machine learning for financial risk. The dataset contains about 285,000 anonymized transactions, with fraud representing only around 0.17 percent. That means predicting every transaction as legitimate gives very high accuracy while detecting no fraud at all.

I explored the data, audited exact duplicates, and divided the modeling data into training, validation, and test sets. I compared ordinary logistic regression, class-weighted logistic regression, and a random forest. Scaling was fitted only on training data inside a Pipeline.

I evaluated precision, recall, F1, Average Precision, and ROC-AUC. Among nine model and threshold combinations, random forest at a threshold of 0.3 had the highest validation F1. I fixed that choice before evaluating the test set. It detected 71 of 95 fraud cases, with 11 false alerts: about 87 percent precision and 75 percent recall.

My main takeaway was that model choice depends on the trade-off between missed fraud and false alerts. This was an educational experiment, not a production system: the dataset is historical, the split is random, and actual business costs were not measured.

## 中文理解提纲

- 问题：从交易中识别欺诈，数据中欺诈极少，单看准确率容易误导。
- 方法：探索数据、检查重复、划分三组数据，对比三种模型；预处理只在训练集学习。
- 选择：在验证集比较九个模型与阈值组合，按 F1 选随机森林、阈值 0.3。
- 结果：测试集 95 笔欺诈抓到 71 笔，漏掉 24 笔，误报正常交易 11 笔。
- 思考：召回率和误报之间存在取舍；没有真实成本数据，不能声称给银行节省了多少钱。
- 局限：随机划分、短期历史数据、测试欺诈样本少；早期 EDA 和建模划分不同。更严格的未来研究需要新数据和时间划分。

## Questions to prepare for

- Why can accuracy be misleading? An always-normal classifier misses all fraud.
- Why three data splits? Fit on training, choose on validation, report on test.
- Why did weighting not automatically improve the model? It increased recall but generated many false alerts.
- Is threshold 0.3 universally best? No; it won the chosen validation F1 comparison on this candidate grid.
- Does ROC-AUC equal fraud detection rate? No; recall at the chosen threshold measures that rate.
- What did AI assistance contribute? It provided stepwise explanations, debugging, and presentation/reproducibility cleanup. I practiced the code and interpreted the trade-offs; I did not implement the underlying algorithms from scratch.
