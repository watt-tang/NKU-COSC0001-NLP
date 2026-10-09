# NYT 中文双栏学术报告

阅读成品：`report.pdf`。论文源码：`main.tex`；英文文献：`references.bib`。

## 重生成图与数据审计

从任意工作目录执行（Windows 当前可用环境）：

```powershell
py -3.13 D:\dasanshangbaogao\ziranyuyanchuli\lab1\report\draw.py
```

在一般环境中也可使用 `python draw.py`。需要 pandas、numpy、matplotlib、scikit-learn；不需要 torch、transformers 或 gensim。脚本仅恢复数据划分、建立分词词表、读取保存结果与绘图，不训练任何分类器或词向量，也不执行 BERT 推理。

脚本输出八组矢量 PDF/PNG 图、六个混淆矩阵 CSV、`class_table.tex` 与 `audit.json`。遇到数值不一致会抛出 AssertionError；不要绕过错误后继续使用图表。测试预测通过完整错例列表与唯一测试正文恢复，Accuracy、Macro-F1、分类报告和分歧 CSV 交叉校验。原实验文件不修改。

## 编译

在本目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
```

或执行 `build.ps1`。已包含编译所需的 PDF 图表，首次重新生成图片可运行 `draw.py`。使用 TeX Live 的 XeLaTeX、CTeX、fontspec、booktabs、natbib；当前源码支持 Windows 字体及 Noto 中文字体回退，不能使用 pdfLaTeX 编译。

## 官方模板与文献来源

- `acl.sty`、`acl_natbib.bst`、`acl_official_example.tex`：直接下载自 [ACL 官方样式仓库](https://github.com/acl-org/acl-style-files)，2026-10-09 获取；样式文件保持原样。使用 `preprint` 选项展示作者与页码。
- BERT、GloVe、传统文本分类基线的书目信息来自 [ACL Anthology](https://aclanthology.org/)，Word2Vec 来自 [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)。`references.bib` 为精简的四条英文文献。
- 本文采用 ACL 双栏模板及中文字体适配，视觉参考紧凑学术论文；不是会议投稿合规声明。

## 关键修正与证据边界

1. 11,519 条原始正文全部通过现有缺失/空白清理；“清洗后”不代表已去重。发现 72 条额外重复正文行；训练/验证、训练/测试、验证/测试分别共享 14、10、1 篇正文。结果只能描述原始划分，不能声称正文无泄漏。
2. 六种方法的测试指标和原结果一致，不修改实验成绩。新增图表由真实 CSV 重绘；原 PNG 混淆矩阵也逐格目视核对。
3. BERT `max_length=64` 是含特殊标记的 WordPiece 序列长度，通常至多保留 62 个正文子词，不是 64 个英文词。没有精确截断率与长度消融结果。
4. 三轮日志耗时和为 7,703.6577 秒，修正原报告 7,704.61 秒；不等同完整运行墙钟时间。
5. 信用卡错例第一句已包含 `travel credit card perks`。原报告对篮球类比的讨论不能证明商业线索全部位于截断后。
6. 二值/词频的 10 个预测分歧含 1 个两者都错；NYT/AG 的 14 个分歧中 11 个 NYT 对、3 个 AG 对。
7. 当前目录没有词向量原文件/模型检查点、完整验证预测、逐样本 OOV 或全方法耗时。OOV 分母独立重算，分子只核对保存计数与比例；近邻词/系数只检查保存文件的内部结构，不能独立恢复原训练。
8. `audit.json` 还保留排除 10 篇训练重叠测试正文的诊断。六种方法在这 10 篇上均预测正确；这不是去重后重新训练的实验，论文不以它替代原测试或严格无泄漏评估。

`audit.json` 保留初次审计的输入 SHA-256、CSV 行列、逐项检查与证据限制。

## PDF 检查

编译后执行 `py -3.13 verify_pdf.py`，检查页数、中文文本提取、核心指标、字体嵌入、引用与溢出日志、正文字符边界，以及仍存在的原始输入文件摘要，并将页面渲染到 `qa/`。需要 PyMuPDF、Pillow 和 Poppler 的 `pdffonts`。当前修订成品为 7 页，无缺字、未解析引用或 Overfull 警告；`qa/verification.json` 记录本次检查及缺失的原始文件。初次实验数据审计通过 254 项检查，八组图均为矢量对象。

已清理编译辅助文件、日志和临时预览。重新编译、绘图或执行检查时，这些中间文件会重新生成；检查脚本需在删除编译日志前运行。
