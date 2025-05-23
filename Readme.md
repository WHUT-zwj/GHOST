# Readme👻

## Abstract

```Stock trend prediction faces significant challenges due to market sentiment influences, long-term dependencies, and time-varying stock correlation. Specifically, (1) Current sentiment analysis methods exhibit insufficient market sentiment quantification and lack dynamic adaptive integration mechanismsfor market fluctuations. (2) Transformer-based models’ quadratic complexity limits long-term stock prediction while lacking finance-specific temporal inductive biases. (3)Traditional temporal tokenization paradigms forcibly merge multi-stock features, weakening stock correlation modeling while dramatically increasing computational costs. To address these constraints, we present GHOST (Gated Hybrid Organization with Sentiment-guided Temporal Mamba and Stock-wise Tokenization Attention). In particular, we leverage GDELT(Global Database of Events, Language and Tone) sentiment quantification through a Hierarchical Sentiment-Gated Layer for dynamic fusion of affective features with trading data. Additionally, Intra-Stock Mamba Selection Layer achieves time-series linear complexity for long-term forecasting by combining a dynamically parameterized shared state space model to provide specialized financial inductive biases. Moreover, our Stock-wise Tokenization Layer converts temporal tokens into stock tokens while preserving data integrity, enabling Inter-Stock Attention Layer to capture stock correlation via attention between stock tokens and reducing attention computation complexity. Experimental results on real-world stock datasets demonstratethe effectiveness of our model```

#### Figure 1: Architecture of GHOST: Sentiment-Gated Mamba with Stock-wise Tokenization for Multi-Stock Prediction.
![390e4f88383d90937a4b054ec4f5498f](https://github.com/user-attachments/assets/b7950514-926d-4674-bbc3-7226863d4470)
#### Figure 2: GDELT-based Sentiment Quantification
![market sentiment](https://github.com/user-attachments/assets/45954747-4a10-4a9a-ab80-03de6f7f3807)
#### Figure 3: Backtesting Pipeline
![image](https://github.com/user-attachments/assets/72f83a98-de58-4503-945a-6eefe70e2211)

## Usage
i.   First, configure the environment according to the requirement.txt file

Ensure that the following libraries have aligned versions:

- causal-conv1d==1.1.0
- mamba-ssm==1.1.1
- torch==2.1.1+cu118
- torchvision==0.16.1+cu118
- torchaudio==0.16.1+cu118

ii. Download stock data to ```.\dataset\stock_data```  

market sentiment data to ```.\dataset```

iii. Finally, ```python run.py``` .

💡💡💡 Don't forget to modify the number of input stocks and the number of features.Considering that ```mamba-ssm``` depends on a Linux environment, we used ```Ubuntu 22.04``` with ```miniconda``` as the basic environment. I have also placed the wheels of these two libraries in the corresponding links, hoping this will make the reproduction of this work more convenient and quick.

❄[Update opensource wheel]:[Baidu](https://pan.baidu.com/s/1-X5RW5o1g5tKViWhtyLjvw?pwd=6666)

❄[Update opensource wheel]:[OneDrive](https://1drv.ms/f/c/fe4981f5f2f28564/EquQIXHxFJFDtZgmVzLUpekB042TPyjfscZ6R4Vvk5BXbw?e=ZTFn9w)

## Dataset


We provide stock data from two markets: ```CSI300``` and ```NASDAQ100```. After data preprocessing, 189 and 64 stocks remain respectively, along with corresponding market sentiment data: ```CHN_NEWS_sentiment.csv``` and ```USA_NEWS_sentiment.csv```.

🔥[Update opensource data]: [Baidu](https://pan.baidu.com/s/1shZ0xDFyGsf5a4h8JgMHxQ?pwd=6666)

🔥[Update opensource data]:[OneDrive](https://1drv.ms/f/c/fe4981f5f2f28564/Ero14-xoBLpHjjc-pBlr19EBRlIDeEmjQ7laLJutptEKEQ?e=U0C70F)

The market sentiment information of China and the United States is time-step aligned with the stock data of CSI300 and NASDAQ100 respectively. You can choose any connection to download the data.

At the same time, you need to modify ```sentiment_path="your root"``` in ```.\data_provider\data_loader``` to adapt to the file path of sentiment data.Also, note that there are some differences in the feature engineering of NASDAQ100 and CSI300 stock data, so you need to modify the ```feature_columns``` in ```.\data_provider\data_loader```accordingly.

🚀Finally, the complete code for market sentiment data will also be open-sourced in the future.<br>

**Build by [Weijie Zhu](https://github.com/WHUT-zwj) and [Jiangling Zhang](https://github.com/WHUT-ZJL)**

