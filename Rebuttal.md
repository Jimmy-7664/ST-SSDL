```
R1. 4，2
1.为什么用weekly 做pattern =》交通数据的weekly pattern很明显
2.运行多次结果
3.Prototype=>真实数据波形
R2. 5，3
1.Anchor如何处理异常值或非平稳值？
2.如何处理非周期数据
3.Prototype=>真实数据波形
R3. 3，2
1.和HimNet比Performance Diss参数量/Training Time
2.和VQ-VAE的对比？
3.除了计算均值，是否有其他的获得anchor策略？HI是最直观简单有效的方法，稳定，可以处理异常值，其他方法futurework
4.dyanamic deviation和distribution shift的区别，解释deviation
R4. 3，3
1.换头实验GCRU=>Trans/Linear等
2.如何捕获空间关系？
3.应用到其他领域的数据集(Climate/Energy/Mobility) 
标题是sp timeseries,主要domain是transportation，由于复杂形是short-term，其他的time series forecasting都是long-term的，96/336，
引一些文章说明常用benchmark，可以扩展到其他领域，引用模型+会议
4.Efficiency 太低由于GCRU
5.符号指代不清X^t和X_\tau
6.Figure4介绍
7.Anchor如何计算？
8.Prototype=>真实数据波形
```

实验：07、08、MLP-SSDL,Transformer-SSDL, Efficiency vs HimNet

# Response to Reviewer hTWY:

We would first like to thank you for the positive comments and valuable feedback. We respond to your comments and questions below.

#### Weakness

> Q1: Let's start from the first weakness. The authors present the work in a very abstract way. An example would have make things clear, and would motivate the work. Fig 1: I couldn;t quite understand.

**A1:** Thank you for the helpful comments. In Figure 1, the “?” highlights our uncertainty about ***how far/different*** the current input should lie from its history in latent space under different cases. This conveys our key idea in two-folds: (1) 我们用history average当做现在input的anchor，用他们之间的distance ($D_1$ $D_2$ in Figure 1) we call this distance as deviation. we use this deviation to boost the forecasting performance; (2) this idea sounds like a typical anomaly detecting method
(2) this requires us to learn to project this deviation from physical space to latent space, the "?" denotes the deviation in latent space here we denote $\widetilde D_1$ $\widetilde D_2$; (3) however, accurately learning  $\widetilde D_1$ $\widetilde D_2$ in continuous latent space is non-trivial.

While these differences are obvious in the input space, it remains unclear how to quantify them in continuous latent space.



Our key idea is **relative distance consistency**: **the current-history pairs that are close (far) in the physical space should remain close (far) in latent space**. i.e. $D_1$ > $D_2$ $\Rightarrow$ $\widetilde D_1$ > $\widetilde D_2$ 



For a concrete illustration, please kindly see **Figure 4** in **Case Study (Sec. 5.6)**:

- **Low deviation** – current and historical curves nearly overlap; their latent representations are also close and select the *same* prototype.
- **Medium deviation** – a medium gap in the physical space corresponding to a moderate latent distance; the two queries are assigned to two *nearby* prototypes.
- **High deviation** – a large physical gap leads to widely separated latent representations and two *distant* prototypes.

In the final version we will make this flow clearer by (i) revising and adding more explanation of Figure 1 and (ii) explicitly stating the principle in Introduction. We hope this could resolve the concerns and making our motivation more clear.

> Q2: The results are extensive but for a very specific type of domains.

**A2:** We thank the reviewer for this insightful suggestion. We agree that evaluating our method's performance across multiple domains would be a valuable extension to our work and would provide stronger evidence of its generalization capabilities. Crucially, the design of ST-SSDL  is **domain-agnostic**. Therefore, the same architecture can be applied to other spatiotemporal domains. Expanding experiments to these domains is an important next step and  we plan to include additional datasets from climate domain in our final version.

Reference (GTS) 我们用的数据集已经是有代表性的时空领域常用的时空benchmark 很多时空领域的模型用这些benchmark测试performance (Model Name, KDD'20, XX citations) diss not widely used

结合2，3

**A2:** We thank the reviewer for this insightful suggestion. We agree that evaluating our proposed ST-SSDL on a more diverse range of spatiotemporal domains would indeed provide a broader perspective on its generalization capabilities. We primarily focus on transportation domain because it is widely regarded as a quintessential and highly challenging benchmark for spatiotemporal prediction due to its complex and dynamic relation [1,2,3,4,5]. We consider the extension of ST-SSDL to these other domains a valuable and promising direction for future work. We will explicitly mention this as a potential avenue for future research in the revised version of our paper.

**A2:** We thank the reviewer for this insightful suggestion. Indeed, our proposed Self-Supervised Deviation Learning framework can be generalized to other spatiotemporal domain such as air quality prediction [1] and weather prediction [2].  We will try to extend our method to more spatiotemporal applications in the future.

![image-20250728130346803](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20250728130346803.png)

#### Question

> Q3: into S non-overlapping weekly ->why weekly? Shouldn´t this be application dependent?

**A3:** Thank you for highlighting this. We adopted weekly partitions due to the widely recognized weekly periodicity in transportation data. As stated in Section 4.1, *"These anchors summarize recurring spatio-temporal patterns and serve as references for deviation modeling."* Yes your suggestion is absolutely correct, this partition length is **application-dependent**: for domains with different temporal patterns, it should be adjusted accordingly. Importantly, ST-SSDL's core design does not inherently depend on weekly periodicity, making such adjustments feasible. We will clearly note this point explicitly in Sec. 4.1 in the final version.

> Q4:  Experiment statistical significance-.> I do not understand tthat reply, can´t you use a fixed number of seeds?

**A4:** Thank you for pointing this out. We have now conducted additional experiments with multiple random seeds (e.g., 5 runs per setting) on four datasets. The results, including mean and standard deviation across these runs, demonstrate that ST-SSDL's performance gains over baseline models remain consistent and statistically significant. We will clearly report these new results in  in the final version.

| Dataset   | Horizon          | Metric | ST-SSDL (Mean ± Std) | ST-SSDL (Paper) |
| --------- | ---------------- | ------ | -------------------- | --------------- |
| METRLA    | Step 3 (15 min)  | MAE    | 2.61±0.03            | 2.64            |
|           |                  | RMSE   | 5.05±0.04            | 5.09            |
|           |                  | MAPE   | 6.68%±0.05           | 6.73%           |
|           | Step 6 (30 min)  | MAE    | 2.97±0.02            | 2.99            |
|           |                  | RMSE   | 6.08±0.04            | 6.12            |
|           |                  | MAPE   | 8.04%±0.09           | 8.12%           |
|           | Step 12 (60 min) | MAE    | 3.38±0.01            | 3.39            |
|           |                  | RMSE   | 7.18±0.03            | 7.18            |
|           |                  | MAPE   | 9.69%±0.13           | 9.81%           |
| PEMSBAY   | Step 3 (15 min)  | MAE    | 1.28±0.02            | 1.3             |
|           |                  | RMSE   | 2.69±0.06            | 2.74            |
|           |                  | MAPE   | 2.65%±0.06           | 2.70%           |
|           | Step 6 (30 min)  | MAE    | 1.59±0.03            | 1.62            |
|           |                  | RMSE   | 3.65±0.07            | 3.69            |
|           |                  | MAPE   | 3.57%±0.09           | 3.63%           |
|           | Step 12 (60 min) | MAE    | 1.88±0.03            | 1.9             |
|           |                  | RMSE   | 4.39±0.09            | 4.42            |
|           |                  | MAPE   | 4.43%±0.11           | 4.48%           |
| PEMSD7(M) | Step 3 (15 min)  | MAE    | 2.04±0.01            | 2.04            |
|           |                  | RMSE   | 3.87±0.03            | 3.88            |
|           |                  | MAPE   | 4.81%±0.05           | 4.82%           |
|           | Step 6 (30 min)  | MAE    | 2.59±0.01            | 2.6             |
|           |                  | RMSE   | 5.22±0.03            | 5.23            |
|           |                  | MAPE   | 6.50%±0.07           | 6.51%           |
|           | Step 12 (60 min) | MAE    | 3.09±0.01            | 3.09            |
|           |                  | RMSE   | 6.39±0.02            | 6.37            |
|           |                  | MAPE   | 8.06%±0.06           | 8.04%           |
| PEMS04    | Average          | MAE    | 18.12±0.08           | 18.2            |
|           |                  | RMSE   | 29.74±0.07           | 29.8            |
|           |                  | MAPE   | 12.38%±0.06          | 12.41%          |
| PEMS07    | Average          | MAE    | 19.21±0.01           | 19.22           |
|           |                  | RMSE   | 32.61±0.06           | 32.64           |
|           |                  | MAPE   | 8.11%±0.00           | 8.11%           |
| PEMS08    | Average          | MAE    | 13.89±0.08           | 13.86           |
|           |                  | RMSE   | 23.17±0.18           | 22.99           |
|           |                  | MAPE   | 9.15%±0.14           | 9.01%           |



> Q5: with dimensions h set to 128, 64, 242 or 32, depending on the dataset." do you have a rule for this choice?

**A5:** Actually, we did not follow a rigid rule. Instead, we select *h* to roughly scale model capacity with dataset size and match the hidden dimensions used by competing methods.

> Q6: figure 5: can you move back from the latent space to the original data?

**A6:** We thank the reviewer for this meaningful suggestion. While Figure 5 visualizes clusters in latent space that each prototype surrounded by its assigned queries, we can also recover the prototype's pattern in the input space by following method:

1. Collect all input sequences assigned to $P_k$ (i.e., those whose query assigned to $P_k$).  
2. Compute the average of these sequences to form the prototype’s representative curve.

Formally, it can be calculated as following equation:
$$
\text{Pattern}_k \;=\; \frac{1}{|\{i: q_i \in P_k\}|}\sum_{q_i \in P_k} X_i
$$
, where ($X_i$) denotes the real input sequence whose query $(q_i)$  assigned to prototype \($P_k$\).

We then visualize the patterns of prototypes and observe several distinct patterns 因为NIPS不让更新图片因此我们描述一些发现的现象:

- **Prototype 7, 12, 14**: Rapidly decreasing curves → sudden‐drop events.
- **Prototype 0, 2, 3, 4, 9, 13, 16, 18**: Relative flat curves → stable conditions.
- **Prototype 1, 5, 6,  **: Increasing curves → growth trends.
- **Prototype 10, 15, 17 **: Gradually decreasing curves → slow‐decline patterns.  

Moreover, these representative pattern of each prototype also **align well with the visualization of Figure 4**.
These demonstrate that ST-SSDL prototypes capture **meaningful spatiotemporal patterns**. We are committed to including this meaningful visualization and analysis in the final version to strengthen interpretability.

# Reviewer DAZY

We are very grateful for your positive recognition and would like to thank you for the thoughtful comments. We respond to your feedback below.

#### Questions

> Q1:  The paper assumes historical averages are reliable anchors, but real-world data often contains outliers or non-stationary trends (e.g., sudden traffic disruptions). Could ST-SSDL integrate robust statistics or outlier detection to improve deviation modeling in such cases?

**A1:** We appreciate the reviewer’s concern. As described in Section 4.1, our anchor is computed as the average over **the whole** training set. This global mean inherently **smooths out** occasional outliers: any single disruption (e.g., an accident-induced outlier) contributes negligibly when aggregated with the full history. 

Note that, anchoring 只是一个可替换的模块，for our task, we consider historical average as a simple yet effective strategy . Of course, robust statistics or outlier detection can be incorporated for better performance. We will add this in future work especially for 更noisy 的application。

> Q2: The experiments focus on traffic data with stable periodic patterns. How might ST-SSDL need to be modified for domains with irregular or aperiodic deviations (e.g., disease spread or social media activity)?

**A2:** Thank you for this insightful point. Actually, our anchoring mechanism is inspired by $HA$ (Historical Average) [DL-traff,...] ,which is always an important baseline almost for every application in time series forecasting domain.

For most spatiotemporal domains like transportation, climate, and energy, they are with clear periodic cycles (daily, weekly, monthly). Here we utilize historical weekly pattern as a simple yet effective anchoring strategy.  Exploration to domains with aperiodic data (e.g., disease spread) poses new challenges especially for our anchoring part. As you suggested, we can incorporate more robust anchoring strategy. We will include such discussion in the final manuscript.

交通数据也有irregular的情况如事故、突发事件，是交通数据预测的challenge。我们的方法恰恰可以通过model deviation来small /big deviation应对这种情况看FIgure4 (c)。

> Q3: Have the authors analyzed whether learned prototypes correspond to interpretable or human-understandable patterns?

**A3:** We thank the reviewer for this meaningful suggestion. While Figure 5 visualizes clusters in latent space that each prototype surrounded by its assigned queries, we can also recover the prototype's pattern in the input space by following method:

1. Collect all input sequences assigned to `P_k` (i.e., those whose query assigned to P_k).  
2. Compute the average of these sequences to form the prototype’s representative curve.

Formally, it can be calculated as following equation:
$$
\text{Pattern}_k \;=\; \frac{1}{|\{i: q_i \in P_k\}|}\sum_{q_i \in P_k} X_i
$$
, where \(X_i\) denotes the real input sequence whose query \(q_i\)  assigned to prototype \(P_k\).

We then visualize the patterns of prototypes and observe several distinct patterns:

- **Prototype 7, 12, 14**: Rapidly decreasing curves → sudden‐drop events.
- **Prototype 0, 2, 3, 4, 9, 13, 16, 18**: Relative flat curves → stable conditions.
- **Prototype 1, 5, 6,  **: Increasing curves → growth trends.
- **Prototype 10, 15, 17 **: Gradually decreasing curves → slow‐decline patterns  

Moreover, these representative pattern of each prototype also **align well with the visualization of Figure 4**.
These patterns demonstrate that ST-SSDL prototypes capture **meaningful spatiotemporal behaviors**. We are committed to including this meaningful visualization and analysis in the final version to strengthen interpretability.

# Reviewer TcCg

Thank you for your positive comments regarding the comparison, connection to VQ-VAE, and explanation on dynamic deviations, as well as your valuable feedback to which we respond below.

#### Weakness

> Q1: The baselines used for comparison are outdated. While the paper references more recent models, these are not included as baselines, e.g., the meta-learning framework HimNet[13] from KDD 2024.

**A1:** Thank you for this suggestion. While STDN (AAAI 2025) is our most recent baseline, we have now added HimNet (KDD 2024) to our comparisons . 

| Dataset   | Horizon          | Metric | HimNet | ST-SSDL (Mean) |
| --------- | ---------------- | ------ | ------ | -------------- |
| METRLA    | Step 3 (15 min)  | MAE    | 2.6    | 2.61           |
|           |                  | RMSE   | 5.02   | 5.05           |
|           |                  | MAPE   | 6.70%  | 6.68%±0.05     |
|           | Step 6 (30 min)  | MAE    | 2.95   | 2.97±0.02      |
|           |                  | RMSE   | 6.06   | 6.08±0.04      |
|           |                  | MAPE   | 8.11%  | 8.04%±0.09     |
|           | Step 12 (60 min) | MAE    | 3.37   | 3.38±0.01      |
|           |                  | RMSE   | 7.22   | 7.18±0.03      |
|           |                  | MAPE   | 9.79%  | 9.69%±0.13     |
| PEMSBAY   | Step 3 (15 min)  | MAE    | 1.27   | 1.28±0.02      |
|           |                  | RMSE   | 2.68   | 2.69±0.06      |
|           |                  | MAPE   | 2.64%  | 2.65%±0.06     |
|           | Step 6 (30 min)  | MAE    | 1.57   | 1.59±0.03      |
|           |                  | RMSE   | 3.6    | 3.65±0.07      |
|           |                  | MAPE   | 3.52%  | 3.57%±0.09     |
|           | Step 12 (60 min) | MAE    | 1.84   | 1.88±0.03      |
|           |                  | RMSE   | 4.32   | 4.39±0.09      |
|           |                  | MAPE   | 4.33%  | 4.43%±0.11     |
| PEMSD7(M) | Step 3 (15 min)  | MAE    | 2.06   | 2.04±0.01      |
|           |                  | RMSE   | 3.94   | 3.87±0.03      |
|           |                  | MAPE   | 4.87%  | 4.81%±0.05     |
|           | Step 6 (30 min)  | MAE    | 2.64   | 2.59±0.01      |
|           |                  | RMSE   | 5.32   | 5.22±0.03      |
|           |                  | MAPE   | 6.64%  | 6.50%±0.07     |
|           | Step 12 (60 min) | MAE    | 3.19   | 3.09±0.01      |
|           |                  | RMSE   | 6.57   | 6.39±0.02      |
|           |                  | MAPE   | 8.30%  | 8.06%±0.06     |
| PEMS04    | Average          | MAE    | 18.14  | 18.12±0.08     |
|           |                  | RMSE   | 29.88  | 29.74±0.07     |
|           |                  | MAPE   | 12.00% | 12.38%±0.06    |
| PEMS07    | Average          | MAE    | 19.21  | 19.21±0.01     |
|           |                  | RMSE   | 32.75  | 32.61±0.06     |
|           |                  | MAPE   | 8.03%  | 8.11%±0.00     |
| PEMS08    | Average          | MAE    | 13.57  | 13.89±0.08     |
|           |                  | RMSE   | 23.22  | 23.17±0.18     |
|           |                  | MAPE   | 8.98%  | 9.15%±0.14     |



On METR-LA and PEMS-BAY, ST-SSDL’s reaches comparable performance against HimNet. On PEMSD7M and PEMS04 datasets, ST-SSDL outperforms HimNet in most metrics. Besides performance, ST-SSDL uses around 40 % fewer parameters and runs 1.5× faster in training time than HimNet. This demonstrates our proposed deviation learning method is parameter-efficient.

> Q2: The Query-Prototype mechanism closely resembles VQ-VAE’s codebook design, with attention-based soft assignment (Equation (3)) replacing hard quantization. The paper lacks a thorough discussion on how this differs from or improves upon VQ-VAE.

**A2:** We sincerely thank the reviewer for their insightful comment. While there is a visual resemblance in using a set of learnable vectors, we would like to summarize several fundamental differences as follows:

1. Task: 
   Our first fundamental difference lies in the core **task**. Our framework is purposed to **quantify the deviation** between current input and its historical anchor. In this context, the prototypes act as a structured reference grid to measure these dynamic changes. This directly contrasts with VQ-VAE, whose primary goal is **data compression and reconstruction**. It aims to represent an input via a single discrete index from its codebook for efficient, generative purposes, a goal entirely different from ours.

2. Assignment Mechanism: 

   To measure the dynamic deviations, our model requires a rich representation. Our **soft, cross attention-based** approach (Equation (3)) produces a representation that is a weighted sum of all prototypes. This preserves the input's complex relationship with multiple underlying patterns. Conversely, VQ-VAE uses **hard quantization**, selecting only the single closest codebook vector. This is inherently a lossy process that discards relational information, which would be detrimental to our goal of precisely measuring deviation.

   Our **mechanism** is a direct consequence of our objective. To measure the dynamic deviations, we apply a cross-attention mechanism to assign each query to create a richer representation. Conversely, VQ-VAE relies on **hard quantization**, selecting only the single closest codebook vector. While effective for compression, this hard assignment mechanism is non-differentiable and can be unstable in training process. Our cross attention-based mechanism provides a smooth and differentiable landscape, enabling stable and effective optimization of our deviation-aware objectives.

In summary, we do not "quantize" the input in the VQ-VAE sense; we use the prototypes to build a deviation-aware representation. We appreciate the reviewer for prompting this comparison and will add a discussion to the revised manuscript to clarify these fundamental distinctions.

> Q3: The use of averaging to generate historical anchors may be overly simplistic. The paper does not explore or compare alternative dynamic anchoring strategies.

**A3:** 

Thank you for this insightful point. Actually, our anchoring mechanism is inspired by $HA$ (Historical Average) [DL-traff,...] ,which is always an important baseline almost for every application in time series forecasting domain. Especially for most spatiotemporal domains (data has clear periodic cycles), we consider historical weekly pattern is a simple yet effective anchoring strategy. Of course, we agree that there should be different anchoring strategies such as daily/weekly/monthly average or even moving average according to applications. Our core contribution, the deviation learning, should be recognized regardless of the anchoring strategy employed. We will add such kind of discussion in Sec.4.1 in the future version.

> Q4: What is the difference between "dynamic deviations" and "distribution shift"? Can you cite literature to explain "dynamic deviations"?

**A4:** We thank the reviewer for this comment. We would like to clarify the distinction between "dynamic deviations"and the well-known problem of "distribution shift".
**Dynamic Deviation **refers to the difference **between a current observation and its corresponding historical average (anchor)**. The term "dynamic" emphasizes that the magnitude of this deviation **varies across different nodes and time points**, as we illustrate in our paper. Our work proposes a method to explicitly model this varying signal to improve forecasting accuracy.

**Distribution Shift**, in contrast, is a classical problem concerning a change in the underlying data distribution $P(X,Y)$ **between the training and testing sets**. This is a well-established challenge in machine learning, where a model trained on one statistical distribution performs poorly when evaluated on another.

Regarding the request for literature on "dynamic deviations," we respectfully clarify that, to the best of our knowledge, our work is the **first to propose deviation learning based on this concept** for spatiotemporal forecasting. Therefore, the idea as presented is a novel contribution of our paper, aiming to establish it as a critical, yet previously overlooked, aspect of spatiotemporal modeling. We will include this discussion in the final manuscript.

#### Questions

> Q5: In Equation (4) and Equation (5), one loss uses MSE while the other uses MAE. What is the rationale behind this choice?

**A5:** We thank the reviewer for this perceptive question, as our loss functions are deliberately chosen for their distinct objectives. We would like to clarify that Equation (4), $L_{Con}$, is not an MSE loss but a **variant of the Triplet Loss** [1] (norm 2 is typically used),  as noted in the **line 155** of our manuscript. Its goal is to achieve latent space discretization by learning inter-prototype separability. Conversely, for $L_{Dev}$ , we choose the Mean Absolute Error (MAE) for its robustness. Since this "distance-of-distances" manner can encounter large discrepancies in cases of high deviation, using an MSE would square these differences and create unstable gradients. The MAE provides a more stable training signal, preventing the model from being overly perturbed by extreme cases. 

# Reviewer  CVvy

Thank you for your detailed comments! Those reviews are really helpful to improve our manuscripts. Below are detailed responses to questions.

#### Weakness

> Q1: The focus of the paper is the introduction of SSDL, which could be applicable in many dynamic prediction tasks. It is not clear how it may perform on different architectures besides GCRU;

**A1**: 

|                  | METRLA | PEMS04 |
| ---------------- | ------ | ------ |
| MLP              |        |        |
| MLP+SSDL         |        |        |
| Transformer      |        |        |
| Transformer+SSDL |        |        |
| GCRU             |        |        |
| Ours (GCRU+SSDL) |        |        |

> Q2: It is not clear how spatial relationships are captured or leveraged in ST-SSDL; 

**A2**: We thank the reviewer for this question and would like to clarify this process. Spatial relationships in ST-SSDL are explicitly captured and leveraged through its **Graph Convolution Recurrent Unit (GCRU) backbone**, as detailed in Section 4.2 of our paper. **GCRU is a spatiotemporal encoder.** The core of this mechanism is the **graph convolution operation** together with gating mechanisms which is formally defined in Equation (6) and Equation (7). This mechanism enables the model to aggregate information from neighboring nodes at each time step, effectively modeling spatial dependencies based on the graph topology.

> Q3: The experiments are all based on traffic datasets. It is unclear how generalizable the proposed method is to other spatio-temporal domains (e.g., climate, mobility, energy)

**A3**: 

> Q4: Although the model is lightweight in model parameters, inference latency is relatively high due to the GCRU backbone. This limits deployment in real-time applications

**A4**: We thank the reviewer for highlighting this trade-off. We agree and **have clearly discussed** this limitation in the **Sec.5.4 Efficiency Study and Limitation section** of our paper. "The model’s inference speed is slightly affected by the GCRU architecture, though this overhead remains acceptable in most practical settings." While the inference latency is a little higher than some non-recurrent baselines due to the iterative GCRU architecture, we believe it is acceptable in most cases. We honestly present such trade-offs, yet believe our work is a valuable contribution in terms of state-of-the-art performance and parameter efficiency.

> Q5: There seem to be inconsistent notations: notation inconsistency: The authors used (e.g., $X^t$ first appeared in line 103 and throughout Section 4) to denote the recent input sequence $X_{\tau-T+1:{\tau}^{'}}$, but this superscript notation was never clearly defined and may be confused with a time index. Since was already used to indicate the current time, a more consistent and intuitive notation might be $X^{\tau}$ or simply $X_{t-T+1:t}$?

**A5**: We thank the reviewer for pointing out this potential notational ambiguity. Our intention for using the superscript 't' (e.g., in $X^t$, $H^t$, $Q^t$) is to create a consistent notation throughout the paper for all variables related to the **current** input, as opposed to the historical anchor ($X^a$, $H^a$, $Q^a$). Because $t$ is already refer to **current**, we use $\tau$ in subscript to denote the time index in Eq. (1) and Eq. (6). 

For clarity, actually we have included a comprehensive notation table in the appendix to clarify this convention. We acknowledge that defining this more explicitly in the main body of the paper would improve clarity, and we will revise the text accordingly in the final version to prevent any confusion with a time index.

> Q6: In Section 5.6, Figure 4 was not really referred or discussed in text.

**A6**: We thank the reviewer for this comment. We acknowledge this mistake. We will revise line **304** from "high deviation scenarios." to "high deviation scenarios **as Figure 4**." 

#### Questions

> Q7: How the anchors were computed in Section 4.1?

**A7**: We would like to clarify the procedure for computing the historical anchors. As described in Section "**History as Self-Supervised Anchor.**" around lines 120 to 125, the historical anchors are computed from the full training sequence. Specifically, the entire training set is partitioned into non-overlapping weekly segments. The historical anchor is then calculated by averaging the values at aligned time steps across all of these weekly segments. This process ensures that for any given input sequence, we can retrieve a timestamp-aligned historical average that serves as a robust, contextual baseline for deviation modeling. 

Note that the weekly anchoring strategy can be replaced by different ones such as daily/weekly/monthly average or even moving average according to applications.

> Q8: Can you provide further semantic interpretation of the prototypes? Do they correspond to interpretable traffic states (e.g., rush hour, low traffic)?

**A8**: We thank the reviewer for this meaningful suggestion. While Figure 5 visualizes clusters in latent space that each prototype surrounded by its assigned queries, we can also recover the prototype's pattern in the input space by following method:

1. Collect all input sequences assigned to `P_k` (i.e., those whose query assigned to P_k).  
2. Compute the average of these sequences to form the prototype’s representative curve.

Formally, it can be calculated as following equation:
$$
\text{Pattern}_k \;=\; \frac{1}{|\{i: q_i \in P_k\}|}\sum_{q_i \in P_k} X_i
$$
, where \(X_i\) denotes the real input sequence whose query \(q_i\)  assigned to prototype \(P_k\).

We then visualize the patterns of prototypes and observe several distinct patterns:

- **Prototype 7, 12, 14**: Rapidly decreasing curves → sudden‐drop events.
- **Prototype 0, 2, 3, 4, 9, 13, 16, 18**: Relative flat curves → stable conditions.
- **Prototype 1, 5, 6 **: Increasing curves → growth trends.
- **Prototype 10, 15, 17 **: Gradually decreasing curves → slow‐decline patterns  

Moreover, these representative pattern of each prototype also **align well with the visualization of Figure 4**.
These patterns demonstrate that ST-SSDL prototypes capture **meaningful spatiotemporal behaviors**. We are committed to including this meaningful visualization and analysis in the final version to strengthen interpretability.

> Q9: Response to your low score at significance .

**A9**: At least, we would like emphasis the core contribution of our work.

一到两句话
