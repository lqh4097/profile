# NuWa: Deriving Lightweight Class-Specific Vision Transformers for Edge Devices

Ziteng Wei1 Qiang He1,2,* Bing Li1 Feifei Chen3,∗ Hai Jin1 Yun Yang2 

1National Engineering Research Center for Big Data Technology and System, Services Computing Technology and System Lab, Cluster and Grid Computing Lab, Huazhong University of Science and Technology 

2 Swinburne University of Technology 3 Deakin University 

{weiziteng,hqiang,lbing,hjin}@hust.edu.cn feifei.chen@deakin.edu.au yyang@swin.edu.au 

## Abstract

Vision Transformers (ViTs) often need to be compressed for deployment on resource-constrained edge devices like drones and smart vehicles. However, existing model compression methods ignore that many edge devices only require the knowledge of specific classes for their applications. As a result, the derived all-class ViTs retain redundant knowledge and perform suboptimally on these classes. We discovered that simply replacing the calibration dataset with class-specific data does not suffice to address this issue, as these methods face two fundamental limitations. First, they overlook the existence of class-detrimental weights, which interfere with specialization, while removing them can improve class-specific performance. Second, the diversity of target classes and resource constraints on edge devices demand numerous customized models. Existing methods are time-consuming and computationally expensive, thus unscalable. In this work, we present NuWa, a cost-efficient method that addresses these challenges by deriving small ViTs from base ViTs for edge devices with specific class requirements. NuWa performs self-knowledge purification to prune class-detrimental weights and efficiently derives compact ViTs through closed-form optimization. Without post-pruning retraining, the derived edge ViTs surpass the base ViT in class-specific accuracy and accelerate inference. Comprehensive experiments demonstrate that NuWa outperforms state-of-the-art training-free pruning methods on class-specific tasks by up to 29.00% in accuracy. Compared with the best-performing trainingdependent pruning method, NuWa achieves a 33.69× pruning speedup and reduces pruning cost by up to 99.83%, with only a 0.61% average accuracy loss. Project Page: https://github.com/CGCL-codes/NuWa 

![](images/0f40b1336bb0086a700d0f601266f1a0a8df0a2fa2d774ea93c19fe513bd9f9e.jpg)



Figure 1. Pruning can sometimes improve class-specific performance. Randomly removing certain neurons from the MLP modules of DeiT-Base unexpectedly increases model accuracy on specific classes, revealing the existence of class-detrimental weights.


## 1. Introduction

Vision Transformers (ViTs) [10] have been widely adopted to facilitate visual services like image recognition [12, 37], object detection [49, 59], and instance segmentation [23, 41]. Enabling real-time visual services for edge devices like smart vehicles and drones is becoming increasingly important in improving the quality of people’s everyday lives [17, 58]. However, most ViTs demand massive computation and storage resources, making deployment on resourceconstrained edge devices a grand challenge [24, 62]. 

To tackle this challenge, many model compression methods have been proposed to adapt ViTs to edge devices [4, 30, 46, 63]. Among them, structured pruning [7, 11] is particularly edge-friendly due to its hardware compatibility and scalability (Sec. 2). However, existing structured pruning methods [44, 46, 54, 60, 62] mainly pursue model size reduction and ignore the fact that in many scenarios, edge devices focus on specific classes and demand only part of the knowledge from the base ViT [57, 65]. For example, a ViT deployed on a smart vehicle typically needs to focus on recognizing objects like pedestrians, vehicles, and traffic signs, while knowledge about flowers or birds is unnecessary. Class-irrelevant knowledge wastes model capacity, which degrades ViTs’ performance on target classes [15]. As shown in Fig. 2(a), this class-agnostic derivation setting fails to provide class-specific models for edge devices, resulting in suboptimal accuracy. 

![](images/fc66481845883e1fabc8805c8bb15c5f2a4ce90cfdcf5fae1bff9130cdb9cd0d.jpg)


![](images/b79caad8268b6d04057f6f8e4517bba5bc89725a21b4b8272dde23092137b8e8.jpg)


![](images/c1f7b6fba8b57dc585184544befe0589189bd5debe4913ff9e59e313e4a45d5b.jpg)



Figure 2. Comparison of model derivation settings for ViTs. (a) Class-agnostic derivation compresses the base ViT without considering class differences, lacking customization for diverse scenarios. (b) Class-specific derivation uses class-specific data for pruning and retraining but fails to remove class-detrimental knowledge and is time-consuming, limiting scalability. (c) NuWa removes class-detrimental weights and formulates pruning as closed-form optimization problems, enabling fast derivation of lightweight and customized edge ViTs.


![](images/686262f19e5ad9a92bd764f46767fa9f179cee69914c394d53f3bcdacd0847d2.jpg)



Figure 3. Existing pruning methods are time-consuming and costly. Experiments are conducted at a pruning rate of 0.50 using randomly selected 25 ImageNet classes to derive one model from DeiT-Base. Cost is based on AWS EC2 g5.48xlarge pricing.


Existing pruning methods can adapt to target classes by replacing the calibration dataset with class-specific data, as illustrated in Fig. 2(b). However, they suffer from two inherent limitations. 1) Neglect of detrimental weights. They mainly remove unimportant weights to minimize the loss of useful knowledge during pruning. However, under the class-specific derivation setting, we observe that some class-detrimental weights are more worth removing, as pruning them can improve the model performance on target classes. We randomly select 25 classes from ImageNet-1K [42] and remove a subset of neurons from the Multi-Layer Perceptron (MLP) modules of DeiT-Base [47]. Interestingly, we find that pruning certain weights can increase the accuracy of DeiT-Base on these 25 classes, as shown in Fig. 1. These detrimental weights cannot be captured by conventional importance metrics [16, 38, 45, 61] (Sec. 3.2). The derived edge ViTs fail to effectively focus on the target classes. 2) High time and computation cost. Different application scenarios involve distinct sets of target classes, and edge devices vary in their available resources [50]. This heterogeneity calls for a large number of customized models with appropriate sizes. Existing pruning methods are timeconsuming, as they require extensive search for pruning configurations and often rely on computationally-expensive retraining to recover model accuracy. These incur considerable computational and time overheads [22], as shown in Fig. 3. Moreover, they lack reusable intermediate results. New target classes and pruning rates require pruning from scratch. For these reasons, existing pruning methods are impractical for large-scale edge deployment. 

To address these challenges, this paper presents NuWa, a novel and cost-efficient method that can derive accurate and lightweight class-specific ViTs from base ViTs through structured pruning without post-pruning retraining. As shown in Fig. 2(c), NuWa first applies self-knowledge purification (SKP) to filter out class-detrimental knowledge from the base ViT $\gamma _ { B }$ . Specifically, NuWa freezes $\gamma _ { B }$ and embeds learnable mask vectors and control factors into its MLP modules to construct a pruning space. Without additional regularization terms, $\gamma _ { B }$ can leverage the original task loss ${ \mathcal { L } } _ { \mathrm { T } }$ and class-specific data $\mathcal { D } _ { \mathcal { S } }$ to identify and prune the class-detrimental weights autonomously. SKP can be viewed as a process of discovering the “free lunch” within ViTs, as it produces a smaller yet more accurate anchor model $\nu _ { A }$ . Since the amount of detrimental knowledge is limited, $\nu _ { A }$ often requires further compression to meet resource constraints. To achieve this efficiently, NuWa formulates the pruning tasks of Multi-Head Attention (MHA) and MLP modules as optimization problems and derives their closed-form solutions, i.e., analytical solutions, to eliminate the need for retraining. Given resource constraints (Sec. 6 in supplementary material (Suppl.)) and target classes, NuWa can rapidly derive class-specific edge ViTs $\nu _ { E }$ with appropriate size from $\gamma _ { B }$ . Our main contributions are as follows: 

• We reveal the existence of class-detrimental knowledge under the class-specific model derivation setting and propose self-knowledge purification (SKP) to identify and prune such detrimental weights. 

• We formulate the pruning of ViT’s MHA and MLP modules as optimization problems and derive their closedform solutions, providing an efficient and principled approach for fast model derivation. 

• We present NuWa, which leverages these innovative strategies to derive small and accurate class-specific ViTs. To the best of our knowledge, it is the first method for deriving class-specific ViTs. 

• Extensive experiments with six models on four datasets show that NuWa effectively derives lightweight classspecific ViTs, outperforming state-of-the-art trainingfree and training-dependent structured pruning methods in class-specific performance and efficiency. 

## 2. Related Works

Model Compression. Various methods have been developed to derive lightweight ViTs, including low-bit quantization [9, 29, 33, 63], knowledge distillation [4, 18, 56], as well as structured [46, 62] and unstructured [6, 30] pruning. However, some of these methods suffer notable drawbacks (Sec. 7 in Suppl.). Low-bit quantization and unstructured pruning often require specialized infrastructure for inference, which limits their applicability to framework-diverse edge devices [7, 55]. Knowledge distillation does not always provide a suitable student model with an appropriate size for initialization, and training a new one from scratch often incurs excessive computational cost [19, 55]. In contrast, structured pruning produces regularly shaped models that are easy to deploy on diverse edge devices and can transfer knowledge from the parameter space of the base ViT for initialization [2, 32]. It offers the generality, feasibility, and portability needed for edge ViT derivation, thus serving as the foundation of NuWa. It is worth noting that NuWa can be integrated with other compression methods for further model compression [13, 39]. 

Structured Pruning for Transformers. There are two types of structured pruning methods, i.e., training-free and training-dependent pruning. 1) Training-free pruning methods [16, 44, 45, 64] prune weights based on importance metrics such as weight magnitude, activation, or gradient (Sec. 8 in Suppl.). Some methods [25, 32, 44] further compute compensation weights to align the pruned and original models, mitigating accuracy loss. However, their objective is to make the pruned models mimic the base model, ignoring the existence of class-detrimental weights (Fig. 1) under the class-specific model derivation setting. They cannot inject class-specific knowledge through training and inherently assume that pruning always leads to accuracy degradation [48]. As a result, the derived models can never surpass the base model on target classes. In contrast, NuWa removes class-detrimental weights through SKP (Sec. 3.2) and elevates the performance of derived models on target classes during pruning. 2) Training-dependent pruning methods [22, 46, 60, 62] also compute importance scores or train sparse masks to guide pruning. Since they introduce retraining after or during pruning to recover accuracy, they can leverage class-specific data to improve the model’s focus on target classes. However, they still fail to remove class-detrimental weights. More importantly, retraining often takes a long time and incurs excessive computational overhead. This makes them impractical for large-scale model deployment on diverse edge devices with different class requirements and resource constraints [3, 36, 43, 50]. Instead, NuWa computes pruned weights directly, enabling effective and fast model derivation. 

![](images/7d95449dcb4166c314c66bd1b3db56c2632f6e08b73127102241efd38aa9dc3e.jpg)



Figure 4. Existing importance metrics fail to improve classspecific accuracy through pruning. The solid lines with standard deviation bands represent the mean accuracy of pruned DeiT-Base on three random sub-tasks (|S|=25) across different pruning rates.


## 3. Methodology

This section presents NuWa in detail, starting with the notations and preliminaries (Sec. 3.1). Then, it presents the details of the two key components of NuWa, i.e., Self-Knowledge Purification (SKP) and Optimization-based Fast Pruning (OFP). SKP constructs a pruning space that enables the model to autonomously identify and prune classdetrimental weights, aiming to improve model performance on target classes and provide an anchor model for subsequent optimization (Sec. 3.2). Next, OFP performs fast pruning with closed-form optimization to efficiently derive lightweight ViTs from the anchor model (Sec. 3.3). An overview of NuWa is illustrated in Fig. 5. 

## 3.1. Notations and Preliminaries

Vision Transformer. A ViT [10] consists of a patch embedding layer, a series of transformer blocks, and a classifier. Each transformer block includes an MHA and MLP module. Given an input $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ with N d-dimensional patch tokens, each head in the MHA module computes an attention A independently before the results are summed: 

$$
\begin{array}{l} \mathcal {A} ^ {(l, h)} = \operatorname{Head} ^ {(l, h)} (\mathbf {X}) = \operatorname{Softmax} \left(\frac {Q K ^ {\top}}{\sqrt {q _ {l}}}\right) V W _ {O} ^ {(l, h) \top} \\ = \operatorname{Softmax} \left(\frac {\mathbf {X} W _ {Q} ^ {(l , h) ^ {\top}} W _ {K} ^ {(l , h)} \mathbf {X} ^ {\top}}{\sqrt {q _ {l}}}\right) \mathbf {X} W _ {V} ^ {(l, h) ^ {\top}} W _ {O} ^ {(l, h) ^ {\top}} \tag {1} \\ \end{array}
$$

![](images/b964febc4e6b4252d3f865f6c20ef02f3e4688069eecf663a2e7f4909ee0f98d.jpg)



Figure 5. Overview of NuWa. 1) Self-Knowledge Purification (SKP): learning binarized masks to identify and prune class-detrimental weights. 2) Optimization-based Fast Pruning (OFP): pruning MHA and MLP modules through closed-form optimization.


where $W _ { \boldsymbol { Q } | K } ^ { ( l , h ) } \in \mathbb { R } ^ { q _ { l } \times d }$ , $W _ { V } ^ { ( l , h ) } \ \in \ \mathbb { R } ^ { v _ { l } \times d }$ and $W _ { \boldsymbol { O } } ^ { ( l , h ) } \in$ $\mathbb { R } ^ { d \times v _ { l } }$ denote the weights of the h-th head in the l-th block. Bias terms are omitted here for clarity. Given the MHA outputs, also denoted by X for simplicity, the MLP module weighted-sums the outputs of all $e _ { l }$ neurons: 

$$
\mathbf {M L P} ^ {(l)} (\mathbf {X}) = \sum_ {i = 1} ^ {e _ {l}} \phi (\mathbf {X} W _ {1} ^ {(l)} [ i ]) \otimes W _ {2} ^ {(l) \top} [ i ] \tag {2}
$$

where $W _ { 1 } ^ { ( l ) } \in \mathbb { R } ^ { e _ { l } \times d }$ and $W _ { 2 } ^ { ( l ) } \in \mathbb { R } ^ { d \times e _ { l } }$ are the weights of the MLP module in the l-th block, $\phi ( \cdot )$ is a nonlinear activation (e.g., GELU), $W [ i ]$ represent the i-th row vectors of $W$ , and ⊗ is the outer product. 

Sub-Tasks. Let $\mathcal { Y } = \{ y _ { 1 } , \cdot \cdot \cdot , y _ { C } \}$ denote the class set of the pre-training dataset for the base ViT $\gamma _ { B }$ . We aim to derive an edge ViT from $\gamma _ { B }$ , denoted as $\nu _ { E }$ , that performs a sub-task of $\gamma _ { B }$ to perceive a subset of Y, denoted as $s \subset \mathcal { V }$ . The corresponding class-specific data is denoted as $\mathcal { D } _ { \mathcal { S } }$ . 

## 3.2. Self-Knowledge Purification

As illustrated in Fig. 1, under the class-specific model derivation setting, there are class-detrimental weights, the removal of which can improve the model’s performance on specific classes. However, as shown in Fig. 4, existing importance evaluation metrics can not be used to identify these weights, even when calculated with class-specific data $\mathcal { D } _ { \mathcal { S } }$ . 

Constructing Pruning Space. Since there is no prior knowledge indicating which weights are class-detrimental or how many of them exist, we propose SKP to automatically filter class-detrimental knowledge by learning binary masks. Specifically, NuWa freezes the base ViT $\gamma _ { B }$ and embeds a learnable mask vector $M ^ { ( l ) } \in \mathbb { R } ^ { e _ { l } }$ and a control factor $\beta ^ { ( l ) } \in \mathbb { R } ^ { 1 }$ before the down-sampling weight $W _ { 2 } ^ { ( l ) }$ in each MLP. Then, NuWa introduces a binarization operation on $M ^ { ( l ) }$ based on $\beta ^ { ( l ) }$ to construct a pruning space for $\gamma _ { B } \mathbf { : }$ : 

$$
M _ {\text { bin. }} ^ {(l)} [ i ] = \left\{ \begin{array}{l l} 1, & \text { if   } M ^ {(l)} [ i ] \geq \operatorname{Sel} _ {\lfloor e _ {l} \times \sigma (\beta^ {(l)}) \rfloor} (M ^ {(l)}) \\ 0, & \text { otherwise } \end{array} \right. \tag {3}
$$

where $\operatorname { S e l } _ { k } ( v )$ returns the k-th largest value in vector v, and $\sigma ( \cdot )$ denotes the sigmoid function. The binarized mask $M _ { \mathrm { b i n . } } ^ { ( l ) }$ then participates in the forward propagation of the MLP module, modifying Eq. (2) as follows: 

$$
\mathbf {M L P} ^ {(l)} (\mathbf {X}) = \sum_ {i = 1} ^ {e _ {l}} \phi (\mathbf {X} W _ {1} ^ {(l)} [ i ]) \otimes (M _ {\text { bin. }} ^ {(l)} [ i ] \cdot W _ {2} ^ {(l) \top} [ i ]) \tag {4}
$$

This design establishes a structured pruning space for $\gamma _ { B }$ , in which setting $M _ { \mathrm { b i n } } ^ { ( l ) } [ i ] = 0$ corresponds to pruning the i-th row of $W _ { 1 } ^ { ( \overline { { l } } ) }$ and the i-th column of $W _ { \gamma } ^ { ( l ) }$ . Following the function-preserving principle [5, 21], $\beta ^ { ( \tilde { l } ) }$ in each MLP module is initialized to 5.0 (σ(5.0) ≈1.0) to ensure consistency between the model before and after the embedding of the mask vectors. NuWa does not apply SKP to the MHA modules at this stage, as empirical observations show that performing SKP only on the MLP modules yields better results (Sec. 9 in Suppl.). 

Knowledge Purification. To enable the model to identify and prune class-detrimental weights through the learning of $\bar { M } ^ { ( l ) }$ and $\beta ^ { ( l ) }$ , NuWa adopts a straight-through estimator (STE) [40, 51] to address the non-differentiability of the binarization operation. Specifically, the gradient of $M _ { \mathrm { b i n . } } ^ { ( l ) }$ enoted as , while t $\bar { G ^ { ( l ) } }$ , is used asner product e gradient serves as $\ddot { M } ^ { ( l ) }$ $\langle M _ { \mathrm { b i n . } } ^ { ( l ) } , G ^ { ( \bar { l } ) } \rangle$ 

the surrogate gradient for $\beta ^ { ( l ) }$ . Subsequently, NuWa uses class-specific data $\mathcal { D } _ { \mathcal { S } }$ to update $\mathcal { M } \mathit { \Pi } = \{ \mathrm { \bar { M } ^ { ( l ) } } \} _ { l = 1 } ^ { L }$ } l =1 and B = {β(l)}Ll= $B = \{ \bar { \beta } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ =1 1 under the supervision of the original visual task loss $\mathcal { L } _ { \mathrm { T } }$ . Finally, NuWa prunes each MLP module of $\gamma _ { B }$ based on $\mathcal { M } _ { \mathrm { b i n . } } ^ { ( l ) }$ . and obtains an anchor model $\gamma _ { A } \mathrm { : }$ : 

$$
W _ {1} ^ {(l)} = W _ {1} ^ {(l)} \left[ \mathcal {I} \left(\mathcal {M} _ {\text { bin. }} ^ {(l)}\right) \right], W _ {2} ^ {(l)} = W _ {2} ^ {(l)} [:, \mathcal {I} \left(\mathcal {M} _ {\text { bin. }} ^ {(l)}\right) ] \tag {5}
$$

$$
\mathcal {I} (v) = \{i \in [ \dim (v) ] \mid v [ i ] \neq 0 \} \tag {6}
$$

Compared with random pruning $( \mathrm { F i g . ~ 1 ) }$ , the anchor model achieves a higher pruning rate and greater class-specific performance improvement (Fig. 6). Since SKP updates no more than 0.05% of the total parameters in $\gamma _ { B }$ and we observe that a smaller batch size enables $\gamma _ { B }$ to perform a more thorough exploration in the pruning space (Fig. 10), the process of obtaining the anchor model is highly efficient. 

## 3.3. Optimization-based Fast Pruning

Since $\nu _ { A }$ may not be able to fully satisfy the resource constraints of edge devices, NuWa formulates the pruning of MHA and MLP modules as optimization problems to derive their closed-form optimal solutions for further pruning. 

Pruning MHA. NuWa prunes the query–key (QK:q) and value–output (VO:v) dimensions of the MHA modules. This design is motivated by two considerations. First, pruning these fine-grained dimensions provides a higher compression potential than pruning entire attention heads. Second, as shown in Eq. (1), the QK and VO dimensions serve as the intermediate dimensions of two weight multiplications (i.e., $W _ { Q K } = W _ { Q } ^ { \top } W _ { K } , W _ { V O } = W _ { V } ^ { \top } W _ { O } ^ { \top } \in \mathring { \mathbb { R } } ^ { d \times d } )$ and are independent of input features X. This allows the pruning results of these dimensions to be shared across all sub-tasks, which effectively reduces redundant computation during large-scale deployment. Take QK dimension as an example, NuWa formulates the pruning of MHA modules as an optimization problem: 

$$
\min _ {W _ {Q} ^ {(l, h) \prime}, W _ {K} ^ {(l, h) \prime}} \| W _ {Q} ^ {(l, h) \top} W _ {K} ^ {(l, h)} - W _ {Q} ^ {(l, h) \prime \top} W _ {K} ^ {(l, h) \prime} \| _ {F} ^ {2} \tag {7}
$$

wher e W (l,h)′ $W _ { \boldsymbol { Q } | K } ^ { ( l , h ) \prime } \in \mathbb { R } ^ { q _ { l } ^ { \prime } \times d }$ denote the pruned weights with pruned QK dimension $q _ { l } ^ { \prime } < q _ { l }$ . This optimization problem essentially becomes a low-rank matrix approximation problem, where the objective is to find a matrix $W _ { Q K } ^ { ( l , h ) \prime }$ with rank less than $q _ { l }$ , such that the Frobenius norm of its difference from W (l,h) $W _ { Q K } ^ { ( l , h ) }$ is minimized. According to the Eckart-Young theorem [8], singular value decomposition (SVD) guarantees this minimization (Sec. 10 in Suppl.). Therefore, NuWa performs MHA pruning through SVD as follows: 

$$
\begin{array}{l} W _ {Q} ^ {(l, h) \prime} = (U _ {Q K} ^ {(l, h)} [:,: q _ {l} ^ {\prime} ] \Sigma_ {Q K} ^ {(l, h)} [: q _ {l} ^ {\prime},: q _ {l} ^ {\prime} ]) ^ {\top} \times \sqrt {q _ {l} ^ {\prime} / q _ {l}} \\ W _ {V} ^ {(l, h) \prime} = (U _ {V O} ^ {(l, h)} [:,: v _ {l} ^ {\prime} ] \Sigma_ {V O} ^ {(l, h)} [: v _ {l} ^ {\prime},: v _ {l} ^ {\prime} ]) ^ {\top} \tag {8} \\ W _ {K} ^ {(l, h) \prime} = V _ {Q K} ^ {(l, h)} [:,: q _ {l} ^ {\prime} ] ^ {\top}, W _ {O} ^ {(l, h) ^ {\prime}} = V _ {V O} ^ {(l, h)} [:,: v _ {l} ^ {\prime} ] \\ \end{array}
$$

where $U _ { Q K } \Sigma _ { Q K } V _ { Q K } ^ { \top } { = } W _ { Q K }$ and $U _ { V O } \Sigma _ { V O } { V _ { V O } ^ { \top } } { = } W _ { V O }$ . NuWa employs a hyperparameter $\rho ,$ which represents the ratio of retained singular value energy to the total energy, to adaptively determine the pruned dimensions $q _ { l } ^ { \prime }$ and $v _ { l } ^ { \prime } .$ . Take the QK dimension as an example. Under the constraint of $\rho ,$ NuWa prunes the $q _ { l }$ until $q _ { l } ^ { \prime }$ satisfies: 

$$
\frac {1}{H _ {l}} \sum_ {h = 1} ^ {H _ {l}} \left(\sum_ {i = 1} ^ {q _ {l} ^ {\prime} - 1} \sigma_ {Q K} ^ {(l, h, i) ^ {2}} / \sum_ {i = 1} ^ {q _ {l}} \sigma_ {Q K} ^ {(l, h, i) ^ {2}}\right) <   \rho \tag {9}
$$

where σQK $\sigma _ { Q K } ^ { ( l , h , i ) }$ $W _ { Q K } ^ { ( l , h ) }$ By adjusting $\rho ,$ NuWa can balance the pruning intensity of MHA and MLP modules. 

Pruning MLP. After pruning the MHA, NuWa prunes the intermediate dimensions of the MLP module to achieve the desired overall pruning rate α. To avoid excessive pruning in certain blocks, which may harm model performance, and to improve the inference efficiency [53] of the derived $\nu _ { E }$ , NuWa prunes all MLP modules of $\nu _ { A }$ to similar sizes whenever possible (Sec. 9 in Suppl.). The pruned intermediate dimension $e _ { l } ^ { \prime }$ of each block is determined as follows: 

$$
e _ {l} ^ {\prime} = \min (e _ {l}, \lfloor (\sum_ {i = 1} ^ {L} e _ {i} - e _ {\text { prune }}) / L \rfloor) \tag {10}
$$

where $e _ { \mathrm { p r u n e } }$ denotes the total number of neurons to be pruned across all MLP modules. After determining $e _ { l } ^ { \prime } ,$ NuWa performs a forward propagation of $\nu _ { A }$ on $\mathcal { D } _ { \mathcal { S } }$ to obtain the mean activation value $\mathbf { \overline { { \mathfrak { a } _ { i } ^ { ( l ) } } } }$ of each neuron in the MLP across all patches. Meanwhile, it samples $K$ images to extract their block-wise activation features $\mathcal { H } ^ { ( l ) } \in$ $\mathbf { \mathbb { R } } ^ { \mathbf { \check { ( } } K N \mathbf { ) } \times e _ { l } }$ . NuWa retains the top $e _ { l } ^ { \prime }$ neurons with the highest activation values in each MLP, whose indices are denoted by $\mathcal { T } _ { r } ^ { ( l ) }$ ). It then solves the following optimization problem: 

$$
\min _ {W _ {2} ^ {(l) \prime}} \| \mathcal {H} ^ {(l)} W _ {2} ^ {(l) \top} - \mathcal {H} ^ {(l)} [ \mathcal {I} _ {r} ^ {(l)} ] W _ {2} ^ {(l) \top} \| _ {F} ^ {2} \tag {11}
$$

NuWa derives a closed-form solution to the above optimization problem to minimize the knowledge loss caused by MLP pruning (Sec. 10 in Suppl.). Based on this solution, the pruning process for the MLP modules is given by: 

$$
W _ {1} ^ {(l) \prime} = W _ {1} ^ {(l)} \left[ \mathcal {I} _ {r} ^ {(l)} \right] \in \mathbb {R} ^ {e _ {l} ^ {\prime} \times d} \tag {12}
$$

$$
W _ {2} ^ {(l) \prime} = W _ {2} ^ {(l)} \mathcal {H} ^ {(l) \top} \mathcal {H} _ {r} ^ {(l)} (\mathcal {H} _ {r} ^ {(l) \top} \mathcal {H} _ {r} ^ {(l)}) ^ {\dagger} \in \mathbb {R} ^ {d \times e _ {l} ^ {\prime}}
$$

where $\mathcal { H } _ { r } ^ { ( l ) } = \mathcal { H } ^ { ( l ) } [ \mathcal { T } _ { r } ^ { ( l ) } ] \in \mathbb { R } ^ { ( K N ) \times e _ { l } ^ { \prime } }$ . Based on the closedform solutions in Eq. (8) and Eq. (12), NuWa can efficiently prune $\nu _ { A }$ to derive the class-specific edge ViT $\gamma _ { E }$ . 

## 4. Experiments and Analysis

Models, Datasets, and Sub-Tasks. Following prior studies [46, 60], we evaluate NuWa with DeiT-B/S/T and 


Table 1. Comparison between NuWa and training-dependent pruning baselines, with the best and second-best accuracies highlighted in bold and underlined, respectively. Subscripts indicate improvements over DeiT-Base. “(FT)” denotes fine-tuning on $\mathcal { D } _ { \mathcal { S } }$ .


<table><tr><td>Method</td><td><eq>S_1/10</eq></td><td><eq>S_2/10</eq></td><td><eq>S_3/10</eq></td><td>Avg.</td><td><eq>S_4/25</eq></td><td><eq>S_5/25</eq></td><td><eq>S_6/25</eq></td><td>Avg.</td><td><eq>S_7/50</eq></td><td><eq>S_8/50</eq></td><td><eq>S_9/50</eq></td><td>Avg.</td></tr><tr><td>DeiT-Base</td><td>71.20</td><td>81.60</td><td>83.00</td><td>78.60</td><td>79.92</td><td>82.56</td><td>80.67</td><td>81.05</td><td>80.48</td><td>79.64</td><td>84.96</td><td>81.69</td></tr><tr><td>DeiT-Base (FT)</td><td>96.60</td><td>99.00</td><td>97.80</td><td><eq>97.80 \uparrow 19.20</eq></td><td>96.40</td><td>96.96</td><td>96.96</td><td><eq>96.77 \uparrow 15.72</eq></td><td>95.00</td><td>93.36</td><td>94.68</td><td><eq>94.35 \uparrow 12.66</eq></td></tr><tr><td>Anchor Model</td><td>95.80</td><td>98.60</td><td>98.60</td><td><eq>97.67 \uparrow 19.07</eq></td><td>94.80</td><td>97.92</td><td>96.72</td><td><eq>96.48 \uparrow 15.43</eq></td><td>95.04</td><td>94.20</td><td>94.92</td><td><eq>94.72 \uparrow 13.03</eq></td></tr><tr><td colspan="13">Pruning Rate = 0.40</td></tr><tr><td>Random</td><td>94.80</td><td>97.00</td><td>97.00</td><td><eq>96.30 \uparrow 17.70</eq></td><td>94.48</td><td>95.12</td><td>95.44</td><td><eq>95.01 \uparrow 13.96</eq></td><td>92.64</td><td>91.32</td><td>92.88</td><td><eq>92.28 \uparrow 10.59</eq></td></tr><tr><td>X-Pruner</td><td>96.20</td><td>98.40</td><td>97.40</td><td><eq>97.33 \uparrow 18.73</eq></td><td><eq>95.44</eq></td><td><eq>97.12</eq></td><td>95.76</td><td><eq>96.11 \uparrow 15.06</eq></td><td>92.88</td><td>92.52</td><td>93.64</td><td><eq>93.01 \uparrow 11.32</eq></td></tr><tr><td>DC-ViT</td><td>75.80</td><td>95.20</td><td>91.00</td><td><eq>87.33 \uparrow 8.73</eq></td><td>89.44</td><td>91.76</td><td>90.24</td><td><eq>90.48 \uparrow 9.43</eq></td><td>88.88</td><td>87.96</td><td>88.16</td><td><eq>88.33 \uparrow 6.64</eq></td></tr><tr><td>RECAP</td><td><eq>96.60</eq></td><td>98.20</td><td><eq>98.80</eq></td><td><eq>97.87 \uparrow 19.27</eq></td><td>95.04</td><td>96.72</td><td>95.36</td><td><eq>95.71 \uparrow 14.66</eq></td><td>93.48</td><td>92.84</td><td>93.80</td><td><eq>93.37 \uparrow 11.68</eq></td></tr><tr><td>MDP</td><td>93.80</td><td>97.00</td><td>96.40</td><td><eq>95.73 \uparrow 17.13</eq></td><td>93.36</td><td>96.16</td><td>94.32</td><td><eq>94.61 \uparrow 13.56</eq></td><td>92.80</td><td>92.12</td><td>93.24</td><td><eq>92.72 \uparrow 11.03</eq></td></tr><tr><td>NuWa</td><td>96.00</td><td><eq>98.60</eq></td><td>98.60</td><td><eq>97.73 \uparrow 19.13</eq></td><td>94.40</td><td><eq>97.04</eq></td><td><eq>96.72</eq></td><td><eq>96.05 \uparrow 15.00</eq></td><td><eq>93.68</eq></td><td><eq>92.88</eq></td><td><eq>94.44</eq></td><td><eq>93.67 \uparrow 11.98</eq></td></tr><tr><td>NuWa (FT)</td><td>97.00</td><td><eq>98.80</eq></td><td><eq>98.80</eq></td><td><eq>98.20 \uparrow 19.60</eq></td><td>96.16</td><td><eq>97.04</eq></td><td><eq>96.72</eq></td><td><eq>96.64 \uparrow 15.59</eq></td><td><eq>93.96</eq></td><td><eq>93.52</eq></td><td><eq>94.80</eq></td><td><eq>94.09 \uparrow 12.40</eq></td></tr><tr><td colspan="13">Pruning Rate = 0.60</td></tr><tr><td>Random</td><td>88.00</td><td>95.00</td><td>92.40</td><td><eq>91.80 \uparrow 13.20</eq></td><td>90.48</td><td>93.36</td><td>91.52</td><td><eq>91.79 \uparrow 10.74</eq></td><td>89.48</td><td>87.64</td><td>89.64</td><td><eq>88.92 \uparrow 7.23</eq></td></tr><tr><td>X-Pruner</td><td>91.20</td><td>95.40</td><td>95.60</td><td><eq>94.07 \uparrow 15.47</eq></td><td><eq>91.58</eq></td><td>93.04</td><td><eq>93.11</eq></td><td><eq>92.58 \uparrow 11.53</eq></td><td><eq>90.24</eq></td><td><eq>88.18</eq></td><td><eq>90.07</eq></td><td><eq>89.50 \uparrow 7.81</eq></td></tr><tr><td>DC-ViT</td><td>68.20</td><td>79.20</td><td>73.80</td><td><eq>73.73 \downarrow 4.87</eq></td><td>70.56</td><td>75.68</td><td>75.84</td><td><eq>74.02 \downarrow 7.02</eq></td><td>69.84</td><td>69.44</td><td>71.32</td><td><eq>70.20 \downarrow 11.49</eq></td></tr><tr><td>RECAP</td><td>89.40</td><td>95.40</td><td>93.80</td><td><eq>92.87 \uparrow 14.27</eq></td><td>88.88</td><td>92.08</td><td>90.24</td><td><eq>90.40 \uparrow 9.35</eq></td><td>88.60</td><td>87.88</td><td>89.16</td><td><eq>88.55 \uparrow 6.86</eq></td></tr><tr><td>MDP</td><td>90.00</td><td>95.80</td><td><eq>95.80</eq></td><td><eq>93.87 \uparrow 15.27</eq></td><td>90.88</td><td><eq>94.80</eq></td><td>91.52</td><td><eq>92.40 \uparrow 11.35</eq></td><td>90.00</td><td>85.00</td><td>90.04</td><td><eq>88.35 \uparrow 6.66</eq></td></tr><tr><td>NuWa</td><td><eq>91.80</eq></td><td><eq>96.20</eq></td><td>94.60</td><td><eq>94.20 \uparrow 15.60</eq></td><td>89.84</td><td>92.00</td><td>92.48</td><td><eq>91.44 \uparrow 10.39</eq></td><td>85.32</td><td>86.20</td><td>85.96</td><td><eq>85.83 \uparrow 4.14</eq></td></tr><tr><td>NuWa (FT)</td><td>95.20</td><td>98.40</td><td>97.80</td><td><eq>97.13 \uparrow 18.53</eq></td><td>95.36</td><td><eq>96.88</eq></td><td><eq>95.60</eq></td><td><eq>95.95 \uparrow 14.90</eq></td><td>92.32</td><td>92.12</td><td>93.44</td><td><eq>92.63 \uparrow 10.94</eq></td></tr></table>

![](images/a05fd1b1eacf7ac5c1ddf69971cd50bd3672434cf7f473e541ac7702f83b23b8.jpg)



Figure 6. Comparison between NuWa and training-free pruning baselines across sub-tasks of different scales. The lightweight models derived by NuWa outperform the base ViT on target classes, which existing methods fail to achieve.


ViT-L/16 [47] on ImageNet-1K [42], CIFAR-100, and CIFAR-10 [26], Fast/Mask R-CNN (Swin-T) [34] on COCO2017 [31]. Each sub-task $\mathcal { S } _ { i } / N$ contains N randomly selected classes from the corresponding dataset. 

Implementations. During SKP, the learning rates for the mask vectors $\mathcal { M }$ and control factors B are set to 0.001 and 0.1, respectively, and the total training steps are $1 0 ^ { 4 }$ with a batch size of 1. AdamW is used as the optimizer. When applying OFP, the retained energy ratio $\rho$ is adaptively determined based on the target pruning rate (Sec. 11 in Suppl.), and the number of calibration samples K is set to 128. GFLOPs is used to reflect device resource constraints and to compute pruning rates. All experiments are conducted on an NVIDIA RTX 4090 GPU. 

Baselines. We first compare NuWa with representative training-free structured pruning methods, including Magnitude Pruning [16], Wanda-sp [2, 45], and Numerical Pruning [44]. Then, we compare NuWa with four state-of-the-art training-dependent structured pruning methods, including X-Pruner [60], DC-ViT [62], RECAP [22], and MDP [46]. The Random method serves as a lower-bound baseline for evaluation. See Sec. 12 in the Suppl. for more details. 

## 4.1. Main Results

Comparison with Training-Free Baselines. We randomly select nine sub-tasks $( S _ { 1 } / 1 0 – S _ { 9 } / 5 0 )$ from ImageNet and compare the accuracy of edge ViTs derived from DeiT-Base by NuWa and training-free baselines under different pruning rates. As shown in Fig. 6, edge ViTs derived by baselines fail to surpass the base ViT because they retain classdetrimental knowledge. In contrast, NuWa eliminates such knowledge through SKP, producing an anchor model that achieves an average pruning rate of 22.61% while improving the average class-specific accuracy by 15.84%. Compared with the base ViT and the best-performing Numerical Pruning, NuWa achieves an accuracy increase of up to 20.60% and 29.00%, respectively, at a pruning rate of 0.60. These results demonstrate that NuWa effectively takes advantage of the “free lunch” within ViTs, deriving smaller yet more accurate class-specific models without retraining. 

Comparison with Training-dependent Baselines. Tab. 1 reports the accuracy of the edge ViTs derived from DeiT-Base by NuWa and training-dependent baselines. Compared with the original DeiT-Base, NuWa achieves average class-specific accuracy improvements of 15.37% and 10.04% at pruning rates of 0.40 and 0.60, respectively, without any retraining. When compared with the best performing baselines, i.e., X-Pruner, RECAP, and MDP, the models derived by NuWa reach 99.35%, 100.03%, and 100.23% of their average accuracy, respectively. With light fine-tuning (10 epochs as in RECAP), NuWa further surpasses them by 2.00%, 2.64%, and 2.82% on average. For efficiency, we simulate a deployment scenario with N sub-tasks and M heterogeneous edge devices per sub-task, requiring MN models. As shown in Tab. 2, NuWa achieves an average speedup of 33.69× over X-Pruner, the best-performing baseline, for a single model with only 0.61% accuracy loss. 


Table 2. Comparison in derivation efficiency between NuWa and training-dependent baselines across different sub-tasks scales. P and T denote pruning and retraining costs. Accuracy is averaged over $S _ { 1 } { - } S _ { 9 }$ in Tab. 1, and cost is based on AWS EC2 g5.48xlarge pricing.


<table><tr><td rowspan="2">Methods</td><td colspan="4">Overhead (GPU Hours)</td><td colspan="4">AWS Cost (N = 50, M = 10)</td><td rowspan="2">Acc. (%)</td></tr><tr><td>|S|=10</td><td>|S|=25</td><td>|S|=50</td><td>Formula</td><td>|S|=10</td><td>|S|=25</td><td>|S|=50</td><td>Avg.</td></tr><tr><td>X-Pruner (P + T)</td><td>0.99</td><td>2.50</td><td>5.01</td><td>MN × (P + T)</td><td>$8063</td><td>$20360</td><td>$40801</td><td>$23074</td><td>93.77</td></tr><tr><td>DC-ViT (P + T)</td><td>2.08</td><td>2.02</td><td>2.13</td><td>MN × (P + T)</td><td>$16939</td><td>$17265</td><td>$17347</td><td>$17184</td><td>80.68</td></tr><tr><td>RECAP (P + T)</td><td>0.89</td><td>2.25</td><td>4.51</td><td>MN × (P + T)</td><td>$7248</td><td>$18324</td><td>$36729</td><td>$20767</td><td>93.13</td></tr><tr><td>MDP (P + T)</td><td>2.26</td><td>5.61</td><td>11.27</td><td>MN × (P + T)</td><td>$18405</td><td>$45688</td><td>$91783</td><td>$51959</td><td>92.95</td></tr><tr><td>NuWa (P1: Eq. (8))</td><td>0.01</td><td>0.01</td><td>0.01</td><td></td><td>$0.16</td><td>$0.16</td><td>$0.16</td><td>$0.16</td><td></td></tr><tr><td>NuWa (P2: Eq. (5), H(l))</td><td>0.06</td><td>0.07</td><td>0.08</td><td>P1 + N × P2</td><td>$48.86</td><td>$57.01</td><td>$65.15</td><td>$57.01</td><td>93.16</td></tr><tr><td>NuWa (P3: Eq. (12))</td><td>2.28e-4</td><td>2.39e-4</td><td>2.50e-4</td><td>+ MN × P3</td><td>$1.86</td><td>$1.95</td><td>$2.04</td><td>$1.95</td><td>(↓ 0.61)</td></tr><tr><td>NuWa (P1 + P2 + P3)</td><td>0.07</td><td>0.08</td><td>0.09</td><td></td><td>$50.88</td><td>$59.12</td><td>$67.35</td><td>$59.12</td><td></td></tr></table>

![](images/4bb5feb7c2aef77a1eff2dc884beb1e4966cea9b6e76c9000849b8ebedde75c2.jpg)


![](images/93131226bf8a2f8759b4f2ce8ccfeb3d3f569709ec75bbb3568640a142dff40b.jpg)


![](images/d8590c37720d76bec3f599c9f9b1292b23e7085c59060f187f9399369a584cf0.jpg)



Figure 7. Performance of NuWa-derived edge ViTs across different base ViTs and datasets, with randomly selected classes in $s .$


Benefiting from reusable pruning results and calibration features that can be shared across different sub-tasks and devices (Sec. 4.2), NuWa significantly reduces the derivation overhead in large-scale deployments. When N=50 and M=10, NuWa reduces the derivation time and cost by up to 99.83% compared with X-Pruner, and by 99.70%, 99.82%, and 99.93% compared with other baselines. 

Generality Across Models and Datasets. To evaluate the generality of NuWa, we extend the experiments beyond DeiT-Base on ImageNet to include DeiT-Small/Tiny and ViT-Large on CIFAR-10 and CIFAR-100, and Fast/Mask R-CNN (Swin-T) on COCO. As shown in Fig. 7, for recognition tasks, the edge ViTs derived from DeiT-Tiny, DeiT-Small, and ViT-Large consistently outperform the corresponding base ViTs on target classes when the pruning rate is below 0.40, 0.50, and 0.60, respectively. We observe that as the model size increases, the pruning rate of the anchor model also increases, indicating a higher proportion of class-detrimental weights. For more complex detection and segmentation tasks, NuWa can also derive models that outperform the base ViT in mAP on target classes when the backbone pruning rate is below 0.40. 

Inference Speedups. We evaluate the speedups of edge ViTs derived by NuWa on two representative devices, i.e., Jetson Orin NX and NVIDIA RTX 4090. Considering the difference in real-world usage, we use a batch size of 1 for Orin NX to simulate sequential real-time requests and a batch size of 256 for RTX 4090 to simulate serverside batch processing. The results are averaged over the nine models from Tab. 1. As shown in Tab. 3, compared with the base ViT, NuWa achieves a speedup of 1.31×- 2.07× on Orin NX and 1.30×-1.92× on RTX 4090, while reducing memory consumption by 26.47%–58.82% and 12.67%–41.63%, respectively. 

![](images/5ee4319b79b3b1e98fb2a390efe76c669a950be185349bb34307c57323441214.jpg)


![](images/8775c7f4dcc61c0bb26fb29318bef8a4b841b24c59aa0302547f096ee521d39b.jpg)



Figure 8. Comparison in feature and probability distributions before and after SKP. The feature distributions are visualized by applying t-SNE to the CLS tokens from the last block, while the bar charts show the average output probability over $\mathcal { D } _ { S / \sharp }$ 5 .


![](images/16a77d81ff3ae5f5c761fd8c870a574b7b7d3c58d24b2cd09ded25c813f0aa25.jpg)



Figure 9. Breakdown of pruning rates and derivation times across sub-tasks of different scales when the overall pruning rate is 0.60.


## 4.2. Analysis

Interpretability of SKP. As shown in Fig. 6, NuWa can use SKP to prune the model while improving its performance on target classes, which seems counterintuitive. To interpret this, we construct a sub-task S with five random classes from CIFAR-10 and apply SKP to DeiT-B finetuned on CIFAR-10. Fig. 8 shows that SKP slightly blurs class boundaries, particularly among non-target classes in $S ^ { c } \left( S ^ { c } = \mathcal { V } \backslash \mathcal { S } \right)$ , but effectively suppresses the output probabilities of classes in $S ^ { c }$ . This indicates that SKP enhances the model’s focus on target classes by reducing misclassification into $S ^ { c }$ . In practice, we further reinforce this effect by assigning large negative bias values (e.g., -100) to nontarget classes in $\nu _ { A } \mathbf { \ ' } _ { \mathbf { S } }$ classifier. This fundamentally prevents $\nu _ { A }$ from misclassifying inputs into classes in $S ^ { c }$ . 

Breakdown of Derivation. Fig. 9 demonstrates the contributions of different modules to the overall pruning rate and the time breakdown during NuWa’s derivation process. 


Table 3. Comparison in computational efficiency between DeiT-Base and edge ViTs derived by NuWa under different α.


<table><tr><td rowspan="2">Methods</td><td colspan="2">Latency (ms)</td><td colspan="2">Throughput (image/s)</td><td colspan="2">Memory (GB)</td><td rowspan="2">#Param(M)</td><td rowspan="2">FLOPs(G)</td></tr><tr><td>Orin NX</td><td>RTX 4090</td><td>Orin NX</td><td>RTX 4090</td><td>Orin NX</td><td>RTX 4090</td></tr><tr><td>DeiT-Base</td><td>45.45</td><td>274.27</td><td>22.00</td><td>933.39</td><td>0.34</td><td>2.21</td><td>86.57</td><td>17.57</td></tr><tr><td>NuWa (<eq>\mathcal{V}_E = \mathcal{V}_A</eq>)</td><td>34.84 (1.31×)</td><td>211.86 (1.30×)</td><td>28.70</td><td>1208.34</td><td>0.25↓26.47%</td><td>1.93↓12.67%</td><td>66.90↓22.72%</td><td>13.60↓22.61%</td></tr><tr><td>NuWa (<eq>\alpha(\mathcal{V}_E) = 0.40</eq>)</td><td>29.70 (1.53×)</td><td>186.01 (1.47×)</td><td>33.67</td><td>1376.27</td><td>0.21↓38.24%</td><td>1.57↓28.96%</td><td>51.75↓40.14%</td><td>10.53↓40.07%</td></tr><tr><td>NuWa (<eq>\alpha(\mathcal{V}_E) = 0.60</eq>)</td><td>22.01 (2.07×)</td><td>142.52 (1.92×)</td><td>45.43</td><td>1796.23</td><td>0.14↓58.82%</td><td>1.29↓41.63%</td><td>34.48↓60.17%</td><td>7.00↓60.16%</td></tr></table>


Table 4. Ablation study of pruning techniques and settings used by NuWa on DeiT-Base with $\alpha = 0 . 6$ .


<table><tr><td>Setting</td><td><eq>\mathcal{S}_{4}/25</eq></td><td><eq>\mathcal{S}_{5}/25</eq></td><td><eq>\mathcal{S}_{6}/25</eq></td><td>Avg</td></tr><tr><td>NuWa</td><td>89.84</td><td>92.00</td><td>92.48</td><td>91.44</td></tr><tr><td>w/o SKP</td><td>69.04</td><td>76.72</td><td>72.32</td><td>72.69↓18.75</td></tr><tr><td>w/o MHA pruning</td><td>74.48</td><td>75.60</td><td>78.08</td><td>76.05↓15.39</td></tr><tr><td>w/o MLP pruning</td><td>4.72</td><td>7.76</td><td>4.00</td><td>5.49↓85.95</td></tr><tr><td>w/o Activation</td><td>84.64</td><td>87.84</td><td>86.88</td><td>86.45↓4.99</td></tr><tr><td>w/o Optimization</td><td>75.20</td><td>76.08</td><td>78.80</td><td>76.69↓14.75</td></tr></table>

For the pruning rate, SKP contributes an average pruning rate of 22.61%. When the overall pruning rate reaches 0.60, OFP contributes 14.67% and 22.72% pruning rates by further prunes MHA and MLP modules, respectively. Most of the pruning occurs in MLPs, which account for about two-thirds of the total parameters in ViTs. The total derivation time consists of three parts, i.e., $\mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } ,$ , and $\mathcal { P } _ { 3 }$ . $\mathcal { P } _ { 1 }$ corresponds to the SVD-based pruning of $W _ { Q K } ^ { ( l ) }$ and $W _ { V O } ^ { ( l ) } ,$ which is data-free and can be reused across different subtasks and pruning rates. $\mathcal { P } _ { 2 }$ includes SKP $( \mathcal { P } _ { 2 } ^ { ( 1 ) } )$ and oneepoch forward propagation on $\mathcal { D } _ { \mathcal { S } }$ for computing activations $a ^ { ( l ) }$ and features $\mathcal { \bar { H } } ^ { ( l ) } ( \mathcal { P } _ { 2 } ^ { ( 2 ) } )$ . This stage dominates the total computation, but can be reused across different pruning rates. $\mathcal { P } _ { 3 }$ prunes MLP based on Eq. (12). Although its results are not reusable, its computational cost is negligible. 

Ablation Study. We validate the necessity of each design in NuWa on $S _ { 4 } / 2 5 – S _ { 6 } / 2 5$ in Tab. 1. The SKP is essential for enabling NuWa to derive small models that outperform the base ViT. As shown in Tab. $^ { 4 , }$ removing SKP prevents the model from focusing on target classes, causing an average accuracy drop of 18.75%. When OFP skips pruning the MHA or MLP modules, accuracy decreases by 15.39% and 85.95%, respectively. This highlights the necessity of balancing their pruning intensities through $\rho .$ Randomly $\mathcal { T } _ { r } ^ { ( l ) }$ instead of using $a ^ { ( l ) }$ lowers accuracy by 4.99%. Replacing the closed-form solution in Eq. (11) with direct removal of low-activation rows in $W _ { 2 } ^ { ( l ) }$ reduces accuracy by 14.75%. This demonstrates the advantage of NuWa in preserving class-specific knowledge. 

Hyperparameter Analysis. We analyze the effect of key hyperparameters on NuWa’s performance. Unless otherwise specified, all experiments are conducted on DeiT-Base with a pruning rate of 0.60, with the sub-tasks $S _ { 4 } – S _ { 6 }$ in Tab. 1. As shown in Fig. 10, a smaller batch size during SKP leads to a higher pruning rate of $\nu _ { A } .$ , enabling $\gamma _ { B }$ to explore the pruning space more thoroughly. Therefore, we set the batch size to 1, which also improves derivation efficiency. Fig. 11 illustrates the effect of different retained energy ratios $\rho$ on the accuracy of NuWa-derived models across different pruning rates $\alpha .$ As α increases, the optimal $\rho$ (marked with asterisks) gradually decreases. This indicates that a larger proportion of pruning should be allocated to the MHA modules. Since $\mathcal { P } _ { 3 }$ is negligible, NuWa allows efficient search for a proper $\rho$ value in practice to determine the optimal pruning configuration. Fig. 12 shows that activation features $\mathcal { H } ^ { ( l ) }$ computed from 128 randomly sampled (Sec. 9 in Suppl.) images are sufficient for NuWa to obtain optimal closed-form solutions of $W _ { 2 } ^ { ( l ) \prime }$ . 

![](images/f0daa5c4b3e088b20da6c66c6bcc9487506cb6d1b7054a0fbcdd15be06293af7.jpg)


![](images/f7f6facca0b03262cd4f7489b324c4187bcba44886c39850053b894d952bb0a3.jpg)



Figure 10. Pruning rate and accuracy of the anchor model $\nu _ { A }$ during SKP under different batch sizes.


![](images/676cc8b3b4b70a6cabf557b31e61a70f938b831be8c0e9a69da4a7ee94d99044.jpg)


![](images/78683260769eaa1109013883bdffb55572b6d7d2d7cf809bef31f0a7ac5f229b.jpg)



Figure 11. Effect of the retained energy ratio $\rho .$



Figure 12. Effect of the number of calibration samples $K$ .


## 5. Conclusion

This paper presented NuWa, a cost-efficient model derivation method that can derive lightweight class-specific ViTs from pre-trained base ViTs for edge devices. Motivated by the insight that removing certain weights elevates targetclass performance, NuWa introduced Self-Knowledge Purification to identify and prune such class-detrimental weights, producing a refined anchor model. Then, NuWa solved optimization problems on the anchor model to directly obtain the closed-form solutions of pruned weights efficiently. Without retraining, NuWa achieved comparable accuracy to computation-intensive structured pruning baselines with the same pruning rate. It significantly reduced the cost of large-scale class-specific model deployment. Extensive experiments with six models on four datasets demonstrated its effectiveness, efficiency, and generality. 

## Acknowledgments

This research was supported by the National Key R&D Program of China under Grant No. 2023YFB4502400. 



[26] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical Report TR-2009, Toronto, ON, Canada, 2009. 6 





through estimator for binary neural networks training. In IEEE/CVF International Conference on Computer Vision, pages 17055–17064, 2023. 4 



## References



[27] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Toronto, ON, Canada, 2009. 1 





[52] Haocheng Xi, Yuxiang Chen, Kang Zhao, Kai Jun Teh, Jianfei Chen, and Jun Zhu. Jetfire: Efficient and accurate transformer pretraining with int8 data flow and per-block quantization. International Conference on Machine Learning, 2024. 1 





[28] Namhoon Lee, Thalaiyasingam Ajanthan, and Philip HS Torr. Snip: Single-shot network pruning based on connection sensitivity. International Conference on Learning Representations, 2018. 2 





[53] Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language model pre-training via structured pruning. International Conference on Learning Representations, 2023. 5 





[1] Mart´ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016. 1 





[29] Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Peng Gao, and Guodong Guo. Q-vit: Accurate and fully quantized low-bit vision transformer. Advances in Neural Information Processing Systems, 35:34451–34463, 2022. 3 





[54] Huanrui Yang, Hongxu Yin, Maying Shen, Pavlo Molchanov, Hai Li, and Jan Kautz. Global vision transformer pruning with hessian-aware saliency. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18547–18557, 2023. 1, 2 





[2] Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Fluctuation-based adaptive structured pruning for large language models. In AAAI Conference on Artificial Intelligence, pages 10865–10873, 2024. 3, 6 





[30] Zhu Liao, Victor Quetu, Van-Tam Nguyen, and Enzo ´ Tartaglione. Can unstructured pruning reduce the depth in deep neural networks? In IEEE/CVF International Conference on Computer Vision, pages 1402–1406, 2023. 1, 3 





[55] Yifei Yang, Zouying Cao, and Hai Zhao. Laco: Large language model pruning via layer collapse. Conference on Empirical Methods in Natural Language Processing, 2024. 3 





[3] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. International Conference on Learning Representations, 2020. 3 





[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft COCO: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. 6 





[56] Zhendong Yang, Zhe Li, Ailing Zeng, Zexian Li, Chun Yuan, and Yu Li. ViTKD: Feature-based knowledge distillation for vision transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1379–1388, 2024. 3 





[4] Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, and Shanghang Zhang. Movekd: Knowledge distillation for vlms with mixture of visual encoders. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19846–19856, 2025. 1, 3 





[32] Gui Ling, Ziyang Wang, and Qingwen Liu. SlimGPT: Layer-wise structured pruning for large language models. Advances in Neural Information Processing Systems, 37: 107112–107137, 2024. 3 





[57] Shuochao Yao and Tarek Abdelzaher. Model compression for edge computing. In Artificial Intelligence for Edge Computing, pages 153–195. Springer, 2023. 1 





[5] Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2net: Accelerating learning via knowledge transfer. International Conference on Learning Representations, 2016. 4 





[33] Shih-Yang Liu, Zechun Liu, and Kwang-Ting Cheng. Oscillation-free quantization for low-bit vision transformers. In International Conference on Machine Learning, pages 21813–21824. PMLR, 2023. 3 





[58] Shengyuan Ye, Jiangsu Du, Liekang Zeng, Wenzhong Ou, Xiaowen Chu, Yutong Lu, and Xu Chen. Galaxy: A resource-efficient collaborative edge ai system for in-situ transformer inference. In IEEE INFOCOM 2024-IEEE Conference on Computer Communications, pages 1001–1010. IEEE, 2024. 1 





[6] Tianlong Chen, Yu Cheng, Zhe Gan, Lu Yuan, Lei Zhang, and Zhangyang Wang. Chasing sparsity in vision transformers: An end-to-end exploration. Advances in Neural Information Processing Systems, 34:19974–19988, 2021. 3, 1 





[34] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. 6, 5 





[59] Hongtian Yu, Yunjie Tian, Qixiang Ye, and Yunfan Liu. Spatial transform decoupling for oriented object detection. In AAAI Conference on Artificial Intelligence, pages 6782– 6790, 2024. 1 





[7] Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. A survey on deep neural network pruning: Taxonomy, comparison, analysis, and recommendations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 1, 3 





[35] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. 5rd International Conference on Learning Representations (ICLR 2017), 2017. 4 





[60] Lu Yu and Wei Xiang. X-pruner: explainable pruning for vision transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24355–24363, 2023. 1, 3, 5, 6 





[8] John S Chipman. “proofs” and proofs of the eckart–young theorem. In Stochastic processes and functional analysis, pages 71–83. CRC Press, 2020. 5 





[36] Zhenyan Lu, Xiang Li, Dongqi Cai, Rongjie Yi, Fangming Liu, Wei Liu, Jian Luan, Xiwen Zhang, Nicholas D Lane, and Mengwei Xu. Demystifying small language models for edge deployment. In 63rd Annual Meeting of the Association for Computational Linguistics, pages 14747–14764, 2025. 3 





[61] Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In IEEE Conference on Computer Vision and Pattern Recognition, pages 9194–9203, 2018. 2 





[9] Dahun Choi and Hyun Kim. GradQ-ViT: Robust and efficient gradient quantization for vision transformers. In AAAI Conference on Artificial Intelligence, pages 16019–16027, 2025. 3 





[37] Zhiying Lu, Chuanbin Liu, Xiaojun Chang, Yongdong Zhang, and Hongtao Xie. DHVT: Dynamic hybrid vision transformer for small dataset recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1 





[62] Hanxiao Zhang, Yifan Zhou, and Guo-Hua Wang. Dense vision transformer compression with few samples. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15825–15834, 2024. 1, 3, 6, 5 





[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations, 2021. 1, 3 





[38] P Molchanov, S Tyree, T Karras, T Aila, and J Kautz. Pruning convolutional neural networks for resource efficient inference. In 5th International Conference on Learning Representations, 2019. 2 





[63] Yunshan Zhong, You Huang, Jiawei Hu, Yuxin Zhang, and Rongrong Ji. Towards accurate post-training quantization of vision transformers via error reduction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1, 3 





[11] Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, and Xinchao Wang. Depgraph: Towards any structural pruning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16091–16101, 2023. 1 





[39] Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Compact language models via pruning and knowledge distillation. Advances in Neural Information Processing Systems, 37:41076–41102, 2024. 3 





[64] Qinqin Zhou, Kekai Sheng, Xiawu Zheng, Ke Li, Xing Sun, Yonghong Tian, Jie Chen, and Rongrong Ji. Training-free transformer architecture search. In IEEE/CVF Conference 





[12] Joshua Fixelle. Hypergraph vision transformers: Images are more than nodes, more than edges. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9751– 9761, 2025. 1 





[40] Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, and Mohammad Rastegari. What’s hidden in a randomly weighted neural network? In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11893–11902, 2020. 4 





on Computer Vision and Pattern Recognition, pages 10894– 10903, 2022. 3 





[13] Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. Advances in Neural Information Processing Systems, 35:4475–4488, 2022. 3 





[41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Seg- ¨ ment anything in images and videos. International Conference on Learning Representations, 2025. 1 





[65] Yan Zhuang, Zhenzhe Zheng, Yunfeng Shao, Bingshuai Li, Fan Wu, and Guihai Chen. Nebula: An edge-cloud collaborative learning framework for dynamic edge environments. In 53rd International Conference on Parallel Processing, pages 782–791, 2024. 1 





[14] Advait Harshal Gadhikar, Sohom Mukherjee, and Rebekka Burkholz. Why random pruning is all we need to start sparse. In International Conference on Machine Learning, pages 10542–10570. PMLR, 2023. 1 





[42] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211–252, 2015. 2, 6 





[15] Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, ¨ Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020. 2 





[43] Moritz Scherer, Luka Macan, Victor JB Jung, Philip Wiese, Luca Bompani, Alessio Burrello, Francesco Conti, and Luca Benini. Deeploy: Enabling energy-efficient deployment of small language models on heterogeneous microcontrollers. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 43(11):4009–4020, 2024. 3 





[16] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in Neural Information Processing Systems, 28, 2015. 2, 3, 6, 1, 5 





[44] Xuan Shen, Zhao Song, Yufa Zhou, Bo Chen, Jing Liu, Ruiyi Zhang, Ryan A Rossi, Hao Tan, Tong Yu, Xiang Chen, et al. Numerical pruning for efficient autoregressive models. In AAAI Conference on Artificial Intelligence, pages 20418– 20426, 2025. 1, 3, 6, 5 





[17] Wei Hao, Zixi Wang, Lauren Hong, Lingxiao Li, Nader Karayanni, AnMei Dasbach-Prisk, Chengzhi Mao, Junfeng Yang, and Asaf Cidon. Nazar: Monitoring and adapting ml models on mobile devices. In 30th International Conference on Architectural Support for Programming Languages and Operating Systems, pages 746–761, 2025. 1 





[45] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In 12th International Conference on Learning Representations, 2023. 2, 3, 6, 1, 5 





[18] Zhiwei Hao, Jianyuan Guo, Kai Han, Yehui Tang, Han Hu, Yunhe Wang, and Chang Xu. One-for-all: Bridge the gap between heterogeneous architectures in knowledge distillation. Advances in Neural Information Processing Systems, 36, 2024. 3 





[46] Xinglong Sun, Barath Lakshmanan, Maying Shen, Shiyi Lan, Jingde Chen, and Jose M Alvarez. MDP: Multidimensional vision model pruning with latency constraint. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20113–20123, 2025. 1, 3, 5, 6 





[19] Changyi He, Yifu Ding, Jinyang Guo, Ruihao Gong, Haotong Qin, and Xianglong Liu. DA-KD: Difficulty-aware knowledge distillation for efficient large language models. In Forty-second International Conference on Machine Learning, 2025. 3 





[47] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J ´ egou. Training ´ data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, pages 10347–10357, 2021. 2, 6 





[20] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. 1 





[48] Cuong Tran, Ferdinando Fioretto, Jung-Eun Kim, and Rakshit Naidu. Pruning has a disparate impact on model accuracy. Advances in Neural Information Processing Systems, 35:17652–17664, 2022. 3 





[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR 2021), 2021. 4 





[49] Xiao Wang, Yu Jin, Wentao Wu, Wei Zhang, Lin Zhu, Bo Jiang, and Yonghong Tian. Object detection using event camera: A moe heat conduction based detector and a new benchmark dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29321–29330, 2025. 





[22] Fatih Ilhan, Gong Su, Selim Furkan Tekin, Tiansheng Huang, Sihao Hu, and Ling Liu. Resource-efficient transformer pruning for finetuning of large models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16206–16215, 2024. 2, 3, 6, 5 





[50] Hao Wen, Yuanchun Li, Zunshuai Zhang, Shiqi Jiang, Xiaozhou Ye, Ye Ouyang, Yaqin Zhang, and Yunxin Liu. AdaptiveNet: Post-deployment neural architecture adaptation for diverse edge environments. In 29th Annual International Conference on Mobile Computing and Networking, pages 1– 17, 2023. 2, 3 





[23] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2989– 2998, 2023. 1 





[51] Xiao-Ming Wu, Dian Zheng, Zuhao Liu, and Wei-Shi Zheng. Estimator meets equilibrium perspective: A rectified straight 





[24] Linyi Jiang, Silvery D Fu, Yifei Zhu, and Bo Li. Janus: Collaborative vision transformer under dynamic network environment. In IEEE INFOCOM 2025-IEEE Conference on Computer Communications, pages 1–10. IEEE, 2025. 1 





[25] Woojeong Kim, Suhyun Kim, Mincheol Park, and Geunseok Jeon. Neuron merging: Compensating for pruned neurons. Advances in Neural Information Processing Systems, 33:585–595, 2020. 3 



# NuWa: Deriving Lightweight Class-Specific Vision Transformers for Edge Devices

Supplementary Material 


Table 5. Comparison in ViT throughput (images/s) on Jetson Orin NX before and after INT8 quantization and magnitude pruning.


<table><tr><td rowspan="2">Methods</td><td>Base ViT</td><td>INT8</td><td>Magnitude</td></tr><tr><td>(GPU)</td><td>(CPU)</td><td>(GPU)</td></tr><tr><td>DeiT-Base</td><td>22.00</td><td><eq>0.29 \downarrow 98.68\%</eq></td><td><eq>21.99 \downarrow 0.05\%</eq></td></tr><tr><td>DeiT-Small</td><td>64.16</td><td><eq>0.67 \downarrow 98.96\%</eq></td><td><eq>63.28 \downarrow 1.37\%</eq></td></tr><tr><td>DeiT-Tiny</td><td>68.87</td><td><eq>1.50 \downarrow 98.48\%</eq></td><td><eq>67.21 \downarrow 2.41\%</eq></td></tr></table>

## 6. Calculation of FLOPs

We adopt GFLOPs, which is widely used to measure the computational cost of model inference, to reflect the resource constraints of edge devices [1]. FLOPs represent the total number of floating-point operations required for a model to perform inference on a single input. For a Vision Transformer (ViT), let the embedding dimension be d; the query–key dimension, value–output dimension, and number of heads in the MHA module of the l-th block be $q _ { l } , v _ { l } .$ , and $H _ { l }$ , respectively; and let the intermediate dimension of the MLP module in the same block be $e _ { l }$ . Given an input $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ consisting of N patch tokens, the total FLOPs of a ViT with L blocks can be formulated as: 

$$
\text { FLOPs } = (2 N d + N ^ {2}) \sum_ {l = 1} ^ {L} H _ {l} (q _ {l} + v _ {l}) + 2 N d \sum_ {l = 1} ^ {L} e _ {l} \tag {13}
$$

## 7. Limitations of Compression Methods

Low-bit Quantization and Unstructured Pruning. As discussed in Sec. 2, quantization and unstructured pruning rely heavily on specific hardware and software architectures to achieve real acceleration. To illustrate this, we apply INT8 quantization [52] and Magnitude Pruning (with a pruning ratio of 0.50) [16], two representative techniques of quantization and unstructured pruning, to compress DeiT-Base, DeiT-Small, and DeiT-Tiny. Then, we measure the inference throughput of compressed ViTs on a widely used edge device, Jetson Orin NX, with a batch size of 1. As shown in Tab. 5, due to the lack of TensorRT support, the INT8-quantized model can only perform inference on the CPU, resulting in a 98.71% decrease in average throughput. Similarly, the model obtained by unstructured pruning shows almost no acceleration, as current frameworks lack libraries optimized for sparse matrix operations. These results indicate that both methods suffer from poor adaptability when deployed on edge devices with highly heterogeneous frameworks and drivers. 

![](images/08fcdd5cda692cee9562f4394280667668c2880ef85599ce86d9bdfe72ff2800.jpg)



Figure 13. Comparison of convergence speed between logit-based knowledge distillation (KD) and random pruning (Pruning) across different pruning rates with DeiT-Base on CIFAR-10.


Knowledge Distillation. Knowledge distillation (KD) focuses on transferring the teacher model’s knowledge to the student model in the feature space and does not provide a good initialization in the parameter space for arbitrary student architectures. As a result, the student model often needs to be trained from scratch, leading to slow convergence and high computational cost. To illustrate this, we compare the convergence of logit-based KD [20] and random structured pruning [14] with DeiT-Base on CIFAR-10 [27]. Specifically, we use student models that share the same architecture as the pruned ones but are randomly initialized. The teacher model is a DeiT-Base pretrained and fine-tuned on CIFAR-10. As shown in Fig. 13, the pruned models converge significantly faster than their KD counterparts, highlighting the importance of parameter-space knowledge transfer. 

Although the above methods have clear limitations, they are orthogonal to NuWa and can be used together when supported by edge devices to achieve further compression. 

## 8. Importance Metrics for Pruning

Magnitude. Magnitude pruning is a simple and widely used pruning approach in model compression [16]. The basic idea is to prune the weights with the smallest magnitudes, assuming that smaller weights contribute less to the model’s output and, thus, can be removed without significantly affecting performance. In the case of pruning according to the magnitude of weights, the importance score is calculated as $\begin{array} { r } { I ( W ) = | W | = \sum _ { i } | w _ { i } | } \end{array}$ . 

Activation. Activation pruning measures the importance of each structure by analyzing its response to input data [6, 45]. The importance score is calculated as $\textstyle I ( W ) = \sum _ { i } a _ { i }$ . Structures with lower average activation values are considered less important for the current task. Since it relies on input data, activation pruning can capture features specific to certain classes, making it well-suited for our class-specific model derivation. 


Table 6. Accuracy and pruning rate (Acc./Rate) of the anchor models under different SKP application settings.


<table><tr><td>Setting</td><td><eq>S_{4}/25</eq></td><td><eq>S_{5}/25</eq></td><td><eq>S_{6}/25</eq></td><td>Avg</td></tr><tr><td>MLP Only</td><td>94.80/25.38</td><td>97.92/21.52</td><td>96.72/18.82</td><td>96.48/21.91</td></tr><tr><td>MHA Only</td><td>81.20/7.38</td><td>82.80/7.13</td><td>80.80/8.87</td><td>81.60/7.79</td></tr><tr><td>MHA + MLP</td><td>93.52/30.05</td><td>95.52/28.01</td><td>94.08/23.54</td><td>94.37/27.20</td></tr></table>

Gradient. During backward propagation, gradients indicate how sensitive the loss function is to changes in each weight [28, 61]. Weights with smaller gradients are considered less important for the model’s predictions and can be pruned with minimal impact on performance. The importance score is calculated as $\begin{array} { r } { I ( W ) = | \frac { \mathcal { L } ( x ) } { \partial W } | = \sum _ { i } | \frac { \mathcal { L } ( x ) } { \partial w _ { i } } | } \end{array}$ . 

Taylor Expansion Approximation. Taylor pruning assumes that a smaller change in the loss value after removing a weight indicates lower importance of that weight for prediction [38, 54]. However, accurately evaluating the importance of all N weights would require N forward propagations to compute the corresponding loss changes, which is computationally expensive. To reduce this overhead, these methods adopt the following approximation based on Taylor expansion: 

$$
\begin{array}{l} I (w) = | \mathcal {L} (x) - \mathcal {L} _ {w = 0} (x) | \\ = \left| \mathcal {L} (x) - \left(\mathcal {L} (x) - \frac {\partial \mathcal {L} (x)}{\partial w} w + R (w)\right) \right| \tag {14} \\ \stackrel {R (w) \approx 0} {\approx} \left| \frac {\partial \mathcal {L} (x)}{\partial w} w \right| \\ \end{array}
$$

This approximation reduces the complexity of evaluating the importance of all weights from $O ( N )$ to O(1). 

## 9. Ablation and Design Justification

Application of SKP. As discussed in Sec. 3.2, NuWa does not apply SKP to the MHA modules. We conduct experiments to justify this design. Specifically, we apply SKP to different modules of DeiT-Base for the three sub-tasks $S _ { 4 } / 2 5 – S _ { 6 } / 2 5$ in Tab. 1 under three settings, i.e., applying SKP only to MLPs (MLP only), only to MHAs (MHA only), and to both (MHA+MLP). As shown in Tab. 6, when SKP is applied to the MHA modules to prune attention heads, the accuracy of the resulting anchor model decreases, regardless of whether SKP is also applied to the MLP modules. This is because SKP fails to control the pruning rate of the base ViT, leading to the loss of class-relevant knowledge. These results indicate that class-detrimental knowledge mainly resides in the MLP modules, while applying SKP to the MHA interferes with the model’s ability to locate and filter such knowledge. This observation is consistent with our SVD-based pruning of the MHA module, since SVD is data-free and suggests that the MHA stores class-agnostic general knowledge. 

![](images/f5409c46a9a1369cc1837d6387d4d453ca6e94ea5d560ef0d0869563e92b3e81.jpg)



Figure 14. Effect of learning rates for M and B on the accuracy and pruning rate of the anchor model.



Table 7. Comparison of different strategies for determining el during Optimized-based Fast Pruning (OFP).


<table><tr><td>Setting</td><td><eq>S_4/25</eq></td><td><eq>S_5/25</eq></td><td><eq>S_6/25</eq></td><td>Avg</td></tr><tr><td colspan="5">Pruning Rate = 0.40</td></tr><tr><td>Uniform</td><td>94.40</td><td>97.04</td><td>96.72</td><td>96.05</td></tr><tr><td>Proportion</td><td>95.57</td><td>96.85</td><td>95.57</td><td>96.00</td></tr><tr><td>Adaptive</td><td>95.49</td><td>96.53</td><td>95.81</td><td>95.94</td></tr><tr><td colspan="5">Pruning Rate = 0.60</td></tr><tr><td>Uniform</td><td>89.84</td><td>92.00</td><td>92.48</td><td>91.44</td></tr><tr><td>Proportion</td><td>88.72</td><td>92.28</td><td>91.65</td><td>90.88</td></tr><tr><td>Adaptive</td><td>88.88</td><td>92.20</td><td>91.57</td><td>90.88</td></tr></table>

Learning Rate of M and B. During SKP, the learning rates of the mask vectors M and control factors B significantly influence how effectively class-detrimental knowledge is filtered. We evaluate different learning rate combinations on DeiT-Base for sub-tasks $S _ { 4 } { - } S _ { 6 }$ , and report the average accuracy and pruning rate of the resulting anchor models $\nu _ { A }$ . As shown in Fig. 14, a larger learning rate for B leads to more thorough filtering of class-detrimental knowledge, while a slightly smaller learning rate for M yields more precise localization of such detrimental weights. We therefore set $\mathrm { L R } _ { \mathcal { M } } = 0 . 0 0 1$ and $\mathrm { L R } _ { B } = 0 . 1$ , which achieve the highest $\nu _ { A }$ with a balanced pruning rate. 

Target Architecture. Before applying OFP, NuWa needs to determine the target model architecture, i.e., the sizes of $q _ { l } , v _ { l } ,$ , and $e _ { l }$ . For the MHA modules, For the MHA modules, NuWa controls ql and vl through the retained energy rate $\rho .$ For the MLP modules, three strategies are considered for determining el: 

• Uniform: pruning el across all blocks to similar sizes to avoid excessive compression in specific MLP modules. 

• Proportion: allocating the total number of pruned neurons $e _ { \mathrm { p r u n e } }$ to each block proportionally based on the original $e _ { l }$ of the anchor model. 

• Adaptive: globally pruning the $e _ { \mathrm { p r u n e } }$ neurons with the 


Table 8. Comparison in model accuracy under different features sampling strategies during OFP.


<table><tr><td rowspan="2">Methods</td><td colspan="2"><eq>S_4/25</eq></td><td colspan="2"><eq>S_5/25</eq></td><td colspan="2"><eq>S_6/25</eq></td><td colspan="2">Avg</td></tr><tr><td>0.40</td><td>0.60</td><td>0.40</td><td>0.60</td><td>0.40</td><td>0.60</td><td>0.40</td><td>0.60</td></tr><tr><td>Random</td><td>94.40</td><td>89.84</td><td>97.04</td><td>92.00</td><td>96.72</td><td>92.48</td><td>96.05</td><td>91.44</td></tr><tr><td>Max-L2</td><td>94.16</td><td>85.36</td><td>95.68</td><td>90.48</td><td>95.92</td><td>90.24</td><td>95.25</td><td>88.69</td></tr></table>

smallest normalized activation values across all blocks. As shown in Tab. 7, when deriving models for $S _ { 4 } – S _ { 6 }$ from DeiT-Base at pruning rates of 0.60 and 0.40, the Uniform strategy consistently achieves the best performance. Considering its superior accuracy and hardware friendliness, we adopt the uniform strategy to determine $e _ { l }$ . 

Sampling Strategy. During OFP, NuWa requires intermediate features $\mathcal { H } ^ { ( \overline { { l } } ) } \in \mathbb { R } ^ { ( \breve { K } N ) \times d }$ from K sampled images to compute the optimal $W _ { 2 } ^ { ( l ) \prime }$ according to Eq. (12). We compare two sampling strategies, i.e., random sampling (Random) and selecting samples with the largest patchtoken L2 norms (Max-L2). As shown in Tab. $^ { 8 , }$ random sampling yields higher model accuracy. Moreover, since Random does not involve sorting operations, its $\mathcal { P } _ { 2 } ^ { ( 2 ) }$ in Fig. 9 is smaller than that of Max-L2. Therefore, NuWa adopts random sampling to get calibration features. 

## 10. Proof

In this section, we prove that Eq. (8) and Eq. (12) are the closed-form solutions to Eq. (7) and Eq. (11), respectively. 

Proof1: Optimal MHA Pruning. Take $W _ { Q } ~ \in ~ \mathbb { R } ^ { q \times d }$ and $W _ { K } \in \mathbb { R } ^ { q \times d } \left( q < d \right)$ as an example, the optimization objective is to find two matrices $W _ { Q } ^ { \prime } \in \mathbb { R } ^ { q ^ { \prime } \times d }$ and $W _ { K } ^ { \prime } \in \mathbb { R } ^ { \check { q } ^ { \prime } \times d }$ $( q ^ { \prime } < q )$ , such that the Frobenius norm of the difference between $\mathbf { \bar { \it W } } _ { Q K } = { \cal W } _ { Q } ^ { \top } { \boldsymbol { W } } _ { K } \in \mathbb { R } ^ { d \times d }$ and $W _ { Q K } ^ { \prime } = W _ { Q } ^ { \prime \top } W _ { K } ^ { \prime } \in$ $\mathbb { R } ^ { d \times d }$ is minimized, i.e., 

$$
\min _ {W _ {Q} ^ {\prime}, W _ {K} ^ {\prime}} \| W _ {Q K} - W _ {Q} ^ {\top} W _ {K} ^ {\prime} \| _ {F} ^ {2} \tag {15}
$$

We first perform a singular value decomposition (SVD) on $W _ { Q K }$ , obtaining: 

$$
W _ {Q K} = U _ {Q K} \Sigma_ {Q K} V _ {Q K} ^ {\top}, U _ {Q K}, V _ {Q K} \in \mathbb {R} ^ {d \times q}
$$

$$
\Sigma_ {Q K} = \operatorname{diag} \left(\sigma_ {1}, \dots , \sigma_ {q}\right) \in \mathbb {R} ^ {q \times q} \tag {16}
$$

$$
\sigma_ {1} \geq \dots \geq \sigma_ {q} \geq 0, \quad \operatorname{rank} (W _ {Q K}) \leq q
$$

For any $W ~ \in ~ \mathbb { R } ^ { d \times d }$ with rank $( W ) ~ \leq ~ q ^ { \prime }$ , let $Y =$ $U _ { Q K } ^ { \top } W \bar { V _ { Q K } } \in \mathbb { R } ^ { q \times q }$ . Since the Frobenius norm $\| \cdot \| _ { F }$ is invariant under left and right multiplication by orthogonal matrices, we have: 

$$
\left\| W _ {Q K} - W \right\| _ {F} = \left\| \Sigma_ {Q K} - Y \right\| _ {F} \tag {17}
$$

$$
\operatorname{rank} (Y) = \operatorname{rank} (W) \leq q ^ {\prime}
$$

Therefore, it suffices to minimize $\| \Sigma _ { Q K } - Y \| _ { F } ^ { 2 }$ over all matrices Y with rank $( Y ) \leq q ^ { \prime }$ . Expanding the above expression, we obtain: 

$$
\begin{array}{l} \| \Sigma_ {Q K} - Y \| _ {F} ^ {2} = \| \Sigma_ {Q K} \| _ {F} ^ {2} + \| Y \| _ {F} ^ {2} - 2 \langle \Sigma_ {Q K}, Y \rangle \\ = \sum_ {i = 1} ^ {q} \sigma_ {i} ^ {2} + \sum_ {i} s _ {i} (Y) ^ {2} - 2 \operatorname{tr} \left(\Sigma_ {Q K} ^ {\top} Y\right) \tag {18} \\ \end{array}
$$

where $s _ { i } ( Y )$ denotes the i-th singular value of $Y$ . Since the von Neumann trace inequality satisfies: 

$$
\operatorname{tr} (\Sigma_ {Q K} ^ {\top} Y) \leq \sum_ {i} \sigma_ {i} s _ {i} (Y) \tag {19}
$$

and equality holds when $Y$ and $\Sigma _ { Q K }$ share the same left and right singular vectors with their singular values aligned in the same order, we have: 

$$
\begin{array}{l} \| \Sigma_ {Q K} - Y \| _ {F} ^ {2} \geq \sum_ {i = 1} ^ {q} \sigma_ {i} ^ {2} + \sum_ {i} s _ {i} (Y) ^ {2} - 2 \sum_ {i} \sigma_ {i} s _ {i} (Y) \\ = \sum_ {i = 1} ^ {q} (\sigma_ {i} - s _ {i} (Y)) ^ {2} \tag {20} \\ \end{array}
$$

Since rank $( Y ) \leq q ^ { \prime }$ implies $s _ { i } ( Y ) = 0$ for all $i > q ^ { \prime }$ , it follows that: 

$$
\begin{array}{l} \left\| \Sigma_ {Q K} - Y \right\| _ {F} ^ {2} \geq \sum_ {i = 1} ^ {q ^ {\prime}} \left(\sigma_ {i} - s _ {i} (Y)\right) ^ {2} + \sum_ {i = q ^ {\prime} + 1} ^ {q} \sigma_ {i} ^ {2} \tag {21} \\ \geq \sum_ {i = q ^ {\prime} + 1} ^ {q} \sigma_ {i} ^ {2} \\ \end{array}
$$

The lower bound is achieved when $s _ { i } ( Y ) = \sigma _ { i }$ for $i \leq q ^ { \prime }$ and Y shares the same singular vectors as $\Sigma _ { Q K }$ . That is: 

$$
Y ^ {\star} = \left[ \begin{array}{c c} \Sigma_ {Q K, q ^ {\prime}} & 0 \\ 0 & 0 \end{array} \right], W ^ {\star} = U _ {Q K} Y ^ {\star} V _ {Q K} ^ {\top}
$$

$$
W _ {Q K} ^ {\prime \star} = W ^ {\star} = U _ {Q K} [:,: q ^ {\prime} ] \Sigma_ {Q K} [: q ^ {\prime},: q ^ {\prime} ] V _ {Q K} [:,: q ^ {\prime} ] ^ {\top} \tag {22}
$$

where $\Sigma _ { Q K , q ^ { \prime } } = { \tt d i a g } ( \sigma _ { 1 } , \cdot \cdot \cdot , \sigma _ { q ^ { \prime } } )$ . Therefore, when $W _ { Q } ^ { \prime }$ and $W _ { K } ^ { \prime }$ are given by: 

$$
W _ {Q} ^ {\prime} = \left(U _ {Q K} [:,: q ^ {\prime} ] \Sigma_ {Q K} [: q ^ {\prime},: q ^ {\prime} ]\right) ^ {\top} \tag {23}
$$

$$
W _ {K} ^ {\prime} = V _ {Q K} [:,: q ^ {\prime} ] ^ {\top}
$$

the quantity $\| W _ { Q K } - W _ { Q } ^ { \prime \top } W _ { K } ^ { \prime } \| _ { F } ^ { 2 }$ attains its minimum value $\textstyle \sum _ { i = q ^ { \prime } + 1 } ^ { q } \sigma _ { i } ^ { 2 }$ . 

□ 

Proof2: Optimal MLP Pruning. For given $W _ { 2 } ~ \in ~ \mathbb { R } ^ { d \times e }$ , $\mathcal { H } \in \mathbb { R } ^ { ( K \bar { N } ) \times e }$ , and $\mathcal { T } _ { r } \in \mathbb { R } ^ { e ^ { \prime } } ( e ^ { \prime } < e )$ , consider the following problem: 

$$
\min _ {W _ {2} ^ {\prime}} \| \mathcal {H} W _ {2} ^ {\top} - \mathcal {H} [ \mathcal {I} _ {r} ] W _ {2} ^ {\prime \top} \| _ {F} ^ {2} \tag {24}
$$

which essentially amounts to finding the least-squares solution of W ′ ∈ Rd×e′ . $W _ { \mathrm { 2 } } ^ { \prime } \in \mathbb { R } ^ { \bar { d } \times e ^ { \prime } }$ 

Write the column vectors of $B = \mathcal { H } W _ { \gimel } ^ { \top } \in \mathbb { R } ^ { ( K N ) \times d }$ as $B = [ b _ { 1 } , \ldots , b _ { d } ] ( b _ { i } \in \mathbb { R } ^ { K N } )$ and those of $W _ { 2 } ^ { \prime \top } \in \mathbb { R } ^ { e ^ { \prime } \times d }$ as $W _ { 2 } ^ { \prime \top } = [ w _ { 1 } , \ldots , w _ { d } ] ( w _ { i } \in \mathbb { R } ^ { e ^ { \prime } } )$ . Then, 

$$
\begin{array}{l} \| \mathcal {H} W _ {2} ^ {\top} - \mathcal {H} [ \mathcal {I} _ {r} ] W _ {2} ^ {\prime \top} \| _ {F} ^ {2} = \| \mathcal {H} _ {r} W _ {2} ^ {\prime \top} - B \| _ {F} ^ {2} \\ = \sum_ {i = 1} ^ {d} \| \mathcal {H} _ {r} w _ {i} - b _ {i} \| _ {2} ^ {2} \tag {25} \\ \end{array}
$$

where $\mathcal { H } _ { r } = \mathcal { H } [ \mathcal { T } _ { r } ] \in \mathbb { R } ^ { K N \times e ^ { \prime } }$ . Hence, the problem decomposes completely column-wise. For any fixed i, we aim to minimize $\| \mathcal { H } _ { r } w _ { i } - b _ { i } \| _ { 2 } ^ { 2 }$ . Let $\mathcal { C } = \mathrm { c o l } ( \mathcal { H } _ { r } ) \subset \mathbb { R } ^ { K N }$ . According to the orthogonal projection theorem, each $b _ { i }$ admits a unique decomposition: 

$$
b _ {i} = \underbrace {P _ {\mathcal {C}} b _ {i}} _ {\in \mathcal {C}} + \underbrace {r _ {i}} _ {\perp \mathcal {C}} \tag {26}
$$

where $P _ { C }$ denotes the orthogonal projection onto $\mathcal { C }$ and $r _ { i }$ is residual vector. For any $w \in \mathbb { R } ^ { e }$ , we have: 

$$
\| \mathcal {H} _ {r} w - b _ {i} \| _ {2} ^ {2} = \| \mathcal {H} _ {r} w _ {i} - P _ {\mathcal {C}} b _ {i} + r _ {i} \| _ {2} ^ {2}
$$

$$
= \| \mathcal {H} _ {r} w - P _ {\mathcal {C}} b _ {i} \| _ {2} ^ {2} + \| r _ {i} \| _ {2} ^ {2} \geq \| r _ {i} \| _ {2} ^ {2} \tag {27}
$$

This shows that mi $\mathbf { 1 } _ { w } \parallel \mathcal { H } _ { r } w - b _ { i } \parallel _ { 2 } ^ { 2 }$ achieves its minimum when $b _ { i }$ is projected onto ${ \mathcal { C } } ,$ and the image of the optimal solution w must be $\mathcal { H } _ { r } w = P c b _ { i }$ . 

By combining the columns together, we obtain the overall optimality condition: 

$$
\mathcal {H} _ {r} W ^ {\star} = P _ {\mathcal {C}} B \tag {28}
$$

Geometrically, this means that each column of B is simultaneously projected onto col $( \mathcal { H } _ { r } )$ . From the necessary and sufficient condition of orthogonal projection, we obtain: 

$$
\mathcal {H} _ {r} ^ {\top} (\mathcal {H} _ {r} W ^ {\star} - B) = 0 \tag {29}
$$

which is precisely the normal equation for the matrix leastsquares problem. It is equivalent to saying that $\mathcal { H } _ { r } W ^ { \star }$ is the orthogonal projection of B onto $\operatorname { c o l } ( \mathcal { H } _ { r } )$ . 

Let $\mathcal { H } _ { r } ^ { \dagger }$ denote the Moore–Penrose pseudoinverse of $\mathcal { H } _ { r } .$ The orthogonal projection operator can then be written as $P c = \mathcal { H } _ { r } \mathcal { H } _ { r } ^ { \dagger }$ . Hence, 

$$
\mathcal {H} _ {r} W ^ {\star} = (\mathcal {H} _ {r} \mathcal {H} _ {r} ^ {\dagger}) B
$$

$$
\Longrightarrow \quad W ^ {\star} = \mathcal {H} _ {r} ^ {\dagger} B + Z, \quad \mathcal {H} _ {r} Z = 0 \tag {30}
$$

Here, $Z$ is arbitrary (it does not affect $\mathcal { H } _ { r } W ^ { \star }$ , and thus yields the same optimality). In particular, the minimumnorm optimal solution is: 

$$
\begin{array}{l} W _ {2} ^ {\prime \star^ {\top}} = W ^ {\star} = \mathcal {H} _ {r} ^ {\dagger} B = \mathcal {H} _ {r} ^ {\dagger} \mathcal {H} W _ {2} ^ {\top} \\ W _ {2} ^ {\prime \star} = W _ {2} \mathcal {H} ^ {\top} \mathcal {H} _ {r} ^ {\dagger \top} = W _ {2} \mathcal {H} ^ {\top} ((\mathcal {H} _ {r} ^ {\top} \mathcal {H} _ {r}) ^ {\dagger} \mathcal {H} _ {r} ^ {\top}) ^ {\top} \tag {31} \\ = W _ {2} \mathcal {H} ^ {\top} \mathcal {H} _ {r} (\mathcal {H} _ {r} ^ {\top} \mathcal {H} _ {r}) ^ {\dagger} \\ \end{array}
$$


Table 9. Hyperparameters for the class-specific model derivation process of NuWa. Here, $\alpha \in ( 0 , 1 )$ denotes the target pruning rate.


<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Self-Knowledge Purification</td></tr><tr><td>Steps</td><td>10000</td></tr><tr><td>Optimizer</td><td>AdamW [35]</td></tr><tr><td>Batch Size</td><td>1</td></tr><tr><td>Learning Rate (LR)</td><td><eq>LR_B=1e-1, LR_M=1e-3</eq></td></tr><tr><td>LR Scheuler</td><td>Constant</td></tr><tr><td>Weight Decay</td><td>0.05</td></tr></table>


Optimization-based Fast Pruning


<table><tr><td>Retained Energy Ratio ρ</td><td>-0.41α3+0.14α2-0.03α+1.0</td></tr><tr><td>#Calibration Samples</td><td>128</td></tr></table>

## Algorithm 1 Block-Uniform Pruning

Input: neuron counts $\{ e _ { l } \} _ { l = 1 } ^ { L } ,$ , prune neuron count eprune 

Output: intermediate size list $\{ e _ { l } ^ { \prime } \} _ { l = 1 } ^ { L }$ 

$N _ { \mathrm { t o t a l } }  \sum _ { l = 1 } ^ { L }$ 

2: $N _ { \mathrm { t a r g e t } }  \lfloor ( \bar { N _ { \mathrm { t o t a l } } } - e _ { \mathrm { p r u n e } } ) / L \rfloor \triangleright$ target neurons count 

3: I ← argsort $( e _ { l } ) \triangleright$ Sort blocks in ascending order 

4: $e _ { l } ^ { \prime } \gets e _ { l }$ for all $l , L _ { \mathrm { c o u n t } }  0$ 

5: for each i in I do 

6: $L _ { \mathrm { c o u n t } }  L _ { \mathrm { c o u n t } } + 1$ 

7: if $e _ { i } \geq N _ { \mathrm { t a r g e t } }$ then 

8: $e _ { i } ^ { \prime }  N _ { \mathrm { t a r g e t } } \triangleright \mathrm { E q . ~ ( 1 0 ) }$ 

9: eprune $ e _ { \mathrm { p r u n e } } - ( e _ { i } - e _ { i } ^ { \prime } )$ 

10: end if 

11: $N _ { \mathrm { t o t a l } }  N _ { \mathrm { t o t a l } } - e _ { i }$ 

12: $N _ { \mathrm { t a r g e t } }  \lfloor ( N _ { \mathrm { t o t a l } } - e _ { \mathrm { p r u n e } } ) / ( L - L _ { \mathrm { c o u n t } } ) \rfloor$ 

13: end for 

14: return $\{ e _ { l } ^ { \prime } \} _ { l = 1 } ^ { L }$ 

The minimum objective value is determined by the projection residual: 

$$
\begin{array}{l} \min _ {W} \| \mathcal {H} _ {r} W - B \| _ {F} ^ {2} = \| (I - \mathcal {H} _ {r} \mathcal {H} _ {r} ^ {\top}) B \| _ {F} ^ {2} \\ = \sum_ {i = 1} ^ {d} \| (I - \mathcal {H} _ {r} \mathcal {H} _ {r} ^ {\top}) b _ {i} \| _ {2} ^ {2} \tag {32} \\ \end{array}
$$

That is, it represents the sum of squared components of B lying in the orthogonal complement of col(Hr). □ 

## 11. Implementation Details

We summarize the hyperparameter settings of NuWa in Tab. 9. In practical implementation, NuWa further optimizes Eq. (10) to strictly enforce: 

$$
\sum_ {l = 1} ^ {L} e _ {l} ^ {\prime} = \left(\sum_ {l = 1} ^ {L} e _ {l}\right) - e _ {\text { prune }} \tag {33}
$$

as described in Algorithm 1. 

For the Swin Transformer [34], since the number of input patch tokens varies with image size, the corresponding GFLOPs are not fixed. Therefore, NuWa adopts the number of parameters (#Param) as the metric to measure the resource constraint and pruning rate. NuWa distributes the total prunable parameter budget proportionally among the MLP modules of each stage in $\nu _ { A } ,$ computes $e _ { \mathrm { p r u n e } }$ for each stage, and determines $e _ { l } ^ { \prime }$ for each block following Algorithm 1. 

Moreover, NuWa ensures that the pruned dimensions $q _ { l } ^ { \prime } ,$ $v _ { l } ^ { \prime } ,$ and $e _ { l } ^ { \prime }$ are multiples of 8, in order to achieve accelerated inference across a wide range of edge devices. 

## 12. Baselines

NuWa is compared with seven baselines implemented based on the open-source code from GitHub. 

• Magnitude Structured Pruning [16]. Given the overall pruning rate $\alpha .$ uniform pruning is applied to the ql, vl, and $e _ { l }$ dimensions in each block. Specifically, dimensions with the smallest L2 norms are pruned by removing the corresponding rows or columns in the weight matrices without post-pruning retraining. 

• Wanda-sp [45]. Wanda-SP is the structured-pruning variant of Wanda. Given the overall pruning rate $\alpha ,$ it uniformly prunes the $q _ { l } , v _ { l } .$ , and $e _ { l }$ dimensions in each block. Considering the massive activations in Transformer forward propagation, Wanda computes importance scores as the product of activation magnitudes and corresponding weight magnitudes. Weights with the lowest scores are pruned by removing the associated rows or columns in the weight matrices. No retraining is performed after pruning. 

• Numerical Pruning [44]. Numerical Pruning estimates the importance of attention heads in MHA and neuron groups in MLP using Newton method. According to the computed importance scores and the overall pruning rate $\alpha ,$ it adaptively prunes $H _ { l }$ and $e _ { l }$ across blocks. After pruning, a compensation weights is calculated for each pruned weight to restore accuracy, without performing any retraining. 

• X-Pruner [60]. X-Pruner applies learnable masks to $H _ { l }$ and $e _ { l }$ in each block, which are optimized via gradient descent with sparsity-inducing regularization added to the loss function. During training, dimensions with smaller mask values are progressively pruned until convergence. After pruning based on the learned sparse masks, X-Pruner performs retraining to recover accuracy. 

• DC-ViT [62]. DC-ViT determines pruning candidates based on each block’s recoverability and the overall pruning rate α. It then removes entire MHA modules from the selected blocks and randomly prunes the ex-

![](images/6ef975342b2f3510342ed366558abbed7261802634ef2648fec909f73e64cd2d.jpg)



Figure 15. Performance of NuWa under class co-occurrence and long-tailed settings.


pansion dimensions $e _ { l }$ of the MLP modules to achieve the target sparsity. After pruning, the model is retrained to restore accuracy. 

• RECAP [22]. RECAP alternates between pruning, finetuning, and updating to reduce memory usage while preserving accuracy. It estimates weight importance via a Taylor-based criterion and prunes $H _ { l }$ and $e _ { l }$ across blocks accordingly. Only important weights are updated with a Fisher-based mask, achieving substantial memory savings without full-model retraining. 

• MDP [46].MDP jointly prunes multiple dimensions of ViTs, including the embedding dimension $( d )$ , the number of attention heads $( H _ { l } )$ , the query–key and value–output dimensions (ql, vl), and the MLP intermediate dimension (el). It formulates pruning as a mixedinteger nonlinear program (MINLP) problem under latency budgets, solved with Hessian-based importance scores and a precomputed latency lookup table (LUT). After pruning, the model is retrained to restore accuracy. 

## 13. Performance under Complex Settings

To account for the complexity of real-world deployment scenarios, we further evaluate NuWa under two challenging settings: class co-occurrence and long-tailed distributions. 

For the class co-occurrence setting, where multiple categories may appear within a single image, we sample multi-label classification sub-tasks of varying scales from COCO2017 and derive edge ViTs from a DeiT-Base model fine-tuned on COCO2017. As shown in the left panel of Fig. 15, at a pruning rate of 0.40, the models derived by NuWa achieve an average precision of 97.13% relative to the base ViT. This result demonstrates the strong generality of NuWa in complex multi-label scenarios. 

For the long-tailed setting, we construct long-tailed variants of each ImageNet sub-task dataset using the following transformation: 

$$
n _ {i} = n \times \mathrm{IF} ^ {(1 - i) / (| \mathcal {S} | - 1)} \tag {34}
$$

where n is the per-class count in the balanced set, $n _ { i }$ is the count for the i-th class after imbalance, |S| is the number of classes in the sub-task, and IF is the imbalance factor (IF=1 means balanced). As shown in the right panel of Fig. 15, for sub-tasks containing 10 classes, the models derived by NuWa maintain stable accuracy across different imbalance levels at a pruning rate of $0 . 4 0 ,$ demonstrating its robustness under long-tailed data distributions. 

## 14. Pseudocode


Algorithm 2 NuWa – Class-Specific Model Derivation


Input: pretrained all-class base ViT $V_{B}$ , sub-task S, class-specific data $D_{S}$ , overall pruning rate $\alpha$ 


Output: lightweight class-specific edge ViT $\nu _ { E }$


1: # Self-Knowledge Purification (SKP)
2: $\mathcal{M} = \{M^{(l)}\}_{l=1}^{L} \leftarrow \{1 \in \mathbb{R}^{e_l}\}_{l=1}^L$ 3: $\mathcal{B} = \{\beta^{(l)}\}_{l=1}^{L} \leftarrow \{5.0\}_{l=1}^L$ 4: Freeze parameters of $V_B$ 5: Embed M and B into the MLP modules of $V_B$ 6: Train M and B on $D_S$ under supervision of $L_T \triangleright M$ and B are involved in the forward propagation according to Eq. (3) and Eq. (4)
7: Prune $V_B$ according to M and B to obtain the anchor model $V_A$ 8: # Optimization-based Fast Pruning
9: Compute $\rho$ based on $\alpha \triangleright$ equation in Tab. 9
10: Compute $\{q_l'\}_{l=1}^L$ and $\{v_l'\}_{l=1}^L$ based on $\rho \triangleright Eq. (9)$ 11: $V_A \leftarrow$ Prune MHA modules of $V_A$ using SVD based on $\{q_l'\}_{l=1}^L$ and $\{v_l'\}_{l=1}^L \triangleright Eq. (8)$ 12: $e_{prune} = ((1 - \alpha)\mathcal{F}(\mathcal{V}_B) - \mathcal{F}(\mathcal{V}_A)) / 2Nd \triangleright \mathcal{F}$ denotes the function that computes GFLOPs or #Params
13: Compute $\{e_l'\}_{l=1}^L$ based on $e_{prune} \triangleright Eq. (10)$ and Algorithm 1
14: $\{a^{(l)}\}_{l=1}^L, \{\mathcal{H}^{(l)}\}_{l=1}^L \leftarrow$ Perform one epoch of forward propagation of $V_A$ on $D_S$ 15: $V_A \leftarrow$ Prune MLP modules of $V_A$ based on $\{e_l'\}_{l=1}^L, \{a^{(l)}\}_{l=1}^L, \text{and } H^{(l)}\}_{l=1}^L$ 

## 15. Limitations and Future Works

A common limitation of NuWa, as well as other pruning methods that do not rely on post-pruning retraining, is that excessive pruning inevitably leads to significant accuracy degradation. As shown in Fig. 6, pruning-only methods, i.e., Random Pruning, Magnitude Pruning, and Wanda-sp suffer sharp accuracy drops when the pruning rate exceeds 20%. Numerical Pruning, which introduces compensation matrices to recover accuracy, maintains stable performance until around 50% pruning. In contrast, NuWa effectively removes class-detrimental weights through SKP, enabling the derived models to outperform the base ViT on target classes even at a pruning rate of 60%. However, when the pruning rate exceeds 70%, noticeable accuracy degradation still occurs. Moreover, as the sub-task size increases (e.g., $| S | > 5 0 )$ , the performance of derived models also declines. 

To address the performance drop under extremely high pruning rates or large-scale sub-tasks, we envision two promising directions. First, lightweight retraining can be introduced to recover accuracy, as NuWa-derived models retain more class-specific knowledge and therefore require less retraining overhead. Second, an offline-trained hypernetwork could be developed to predict weight updates, enabling the online derivation process to efficiently map task specifications, such as model architecture, sub-task, and pruning rate, to corresponding pruned weights. 