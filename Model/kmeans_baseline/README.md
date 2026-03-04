# kmeans_baseline

对 `Model/data` 中的数据集做 KMeans 聚类基线，并用 `FinalModel/utils.py` 的匈牙利匹配 ACC/NMI/ARI/F1 评估。

## 运行

从仓库根目录运行（推荐使用你的环境）：

```powershell
C:\Users\Miku12\.conda\envs\ahgfc\python.exe Model\kmeans_baseline\main.py --dataset cora
```

常用参数：

```powershell
C:\Users\Miku12\.conda\envs\ahgfc\python.exe Model\kmeans_baseline\main.py `
  --dataset citeseer `
  --seed 1 --repeats 5 `
  --pca_dim 256 `
  --l2_normalize 1
```

PubMed / Amazon(computers) 小图（会写入 extracted_root 下的 *.npz 缓存）：

```powershell
C:\Users\Miku12\.conda\envs\ahgfc\python.exe Model\kmeans_baseline\main.py `
  --dataset pubmed --pubmed_use_small 1 --pubmed_small_n 8000
```

```powershell
C:\Users\Miku12\.conda\envs\ahgfc\python.exe Model\kmeans_baseline\main.py `
  --dataset amazon_electronics_computers --amazon_computers_use_small 1 --amazon_computers_small_n 8000
```

## 输出

每次运行会在 `Model/kmeans_baseline/runs/<dataset>/<tag>/kmeans_results.json` 写入参数、每次重复的指标以及均值/方差汇总。
