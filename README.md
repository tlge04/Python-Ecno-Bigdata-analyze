# 电商用户行为大数据分析期末项目

本项目是“Python 经济大数据分析”课程期末大作业，主题为：

**基于电商用户行为大数据的消费转化漏斗与购买预测分析**

项目使用 Kaggle 数据集 `mkechinov/ecommerce-behavior-data-from-multi-category-store` 中 2019 年 10 月、11 月、12 月三个月用户行为日志，共处理 `177,493,621` 条记录，围绕浏览、加购、购买三阶段漏斗分析电商消费转化，并构建两个机器学习模型：

- 用户-月份购买倾向识别
- 首次加购后的购物车转化预测

## 项目亮点

- 使用 Python 分块读取压缩 CSV，避免一次性加载亿级日志。
- 使用 Pandas / NumPy 完成清洗、聚合、特征工程和逻辑回归建模。
- 分析浏览、加购、购买漏斗，以及月度、品类、品牌、价格带结构。
- 新增购物车转化预测：只使用首次加购前/加购时信息预测后续购买，按 10-11 月训练、12 月测试。
- 使用 TopK / Lift 指标解释模型在营销排序中的业务价值。

## 关键结果

- 总记录数：`177,493,621`
- 浏览事件：`167,321,576`
- 加购事件：`7,350,209`
- 购买事件：`2,821,836`
- 购物车转化样本：`216,713` 个加购 session
- 购物车转化模型 12 月测试集 Top 10%：
  - Precision@Top10%：`60.22%`
  - Lift@Top10%：`1.25`

## 目录结构

```text
.
├── scripts/
│   ├── ecommerce_pipeline.py
│   └── cart_conversion_pipeline.py
├── data/
│   └── processed/ecommerce/
├── outputs/
│   └── figures/
├── 最终使用tex/
│   ├── report/
│   │   ├── ecommerce_report.tex
│   │   └── figures/
│   ├── ppt/
│   │   ├── ecommerce_beamer.tex
│   │   └── figures/
│   └── 项目说明给队友.md
├── 课程简介.md
├── 期末演示要求.md
└── README.md
```

## 数据说明

原始数据没有放入 GitHub。请从 Kaggle 下载并放到：

```text
data/raw/ecommerce/
```

期望文件名：

```text
2019-Oct.csv.gz
2019-Nov.csv.zip
2019-Dec.csv.gz
```

## 运行方式

先运行主分析流水线：

```bash
python scripts/ecommerce_pipeline.py
```

再运行购物车转化预测流水线：

```bash
python scripts/cart_conversion_pipeline.py
```

脚本会输出聚合表、模型指标和图表到：

```text
data/processed/ecommerce/
outputs/figures/
```

## 最终材料

最终报告和展示稿位于：

```text
最终使用tex/report/ecommerce_report.tex
最终使用tex/ppt/ecommerce_beamer.tex
```

报告使用 `USTCReport` 模板，Beamer 使用 USTC Beamer 模板。实际编译建议在 Overleaf 中完成。

## 口径说明

- 用户漏斗中的唯一用户数使用确定性抽样估计。
- 用户-月份模型是购买倾向识别，不夸大为严格实时预测。
- 购物车转化模型以 `user_session` 为样本，只使用首次加购之前和首次加购当时的信息。
- 高预测分数表示更可能自然购买，不等于最需要发券；低自然转化但高商品价值的 session 更适合作为召回候选。

