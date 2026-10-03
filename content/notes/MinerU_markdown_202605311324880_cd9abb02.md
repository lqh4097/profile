# Multi-Objective Neural Architecture Search for Cognitive Diagnosis: Balancing Accuracy and Interpretability With Probabilistic Models

Yifei Sun , Mingkai Duan , Sicheng Hou , Shi Cheng , Maoguo Gong , Fellow, IEEE, and Zhi-Hui Zhan , Fellow, IEEE 

Abstract—Cognitive diagnosis (CD) is a fundamental task within computational social systems, essential for personalizing learning in intelligent education by assessing a learner’s fine-grained knowledge proficiency. A critical dilemma exists in designing cognitive diagnosis models (CDMs): conventional psychometric models are interpretable but often simplistic, while complex deep learning models achieve high accuracy at the cost of becoming uninterpretable “opaque models.” This lack of transparency is a major barrier to trust and adoption in real-world educational social systems. To address this challenge, this article introduces probabilistic model-building for cognitive architecture search (PMB-CAS), a novel framework that automates the discovery of CDMs that balance diagnostic accuracy with structural interpretability. We reframe the design process as a multi-objective optimization problem, balancing diagnostic performance [area under the curve (AUC)] with a refined, operator-weighted model interpretability score (MIS) that fairly assesses topological complexity. The framework navigates a flexible, tree-based search space using a probabilistic modelbuilding strategy to efficiently discover promising architectures. Extensive experiments on benchmark datasets demonstrate that 

Received 12 November 2025; revised 24 February 2026 and 26 April 2026; accepted 27 April 2026. This work was supported in part by the Shaanxi Education Teaching Reform Research Program under Grant 23BY028; in part by the Natural Science Basic Research Plan in Shaanxi Province of China under Grant 2022JM-381; in part by the Shaanxi Normal University Key Program of Teaching Reform Research under Grant 22JG002; in part by the Cooperation Program for High-end Foreign Experts in Ministry of Science and Technology under Grant G2021173001L; in part by the National Natural Science Foundation of China under Grant 61703256 and Grant 62036006; in part by the Fundamental Research Funds for the Central Universities; and in part by the Australian Research Council (ARC) under Grant LP180100114 and Grant DP200102611. (Corresponding author: Yifei Sun.) 

Yifei Sun, Mingkai Duan, and Sicheng Hou are with the School of Physics and Information Technology, Shaanxi Normal University, Xi’an 710119, China (e-mail: yifeis@snnu.edu.cn; 20241905@snnu.edu.cn; sicheng_hou@foxmail.com). 

Shi Cheng is with the School of Computer Science, Shaanxi Normal University, Xi’an 710119, China (e-mail: cheng@snnu.edu.cn). 

Maoguo Gong is with the School of Electronic Engineering, Key Laboratory of Collaborative Intelligence Systems, Ministry of Education, Xidian University, Xi’an 710071, China, and also with the Academy of Artificial Intelligence, College of Mathematics Science, Inner Mongolia Normal University, Hohhot 010011, China (e-mail: gong@ieee.org). 

Zhi-Hui Zhan is with the College of Artificial Intelligence, Nankai University, Tianjin 300350, China (e-mail: zhanapollo@163.com). 

Digital Object Identifier 10.1109/TCSS.2026.3689070 

PMB-CAS discovers a portfolio of Pareto-optimal models that not only achieve state-of-the-art accuracy, outperforming established baselines, but also offer varying degrees of structural transparency. This work provides educators and stakeholders with a spectrum of trustworthy solutions and establishes a new paradigm for developing effective and verifiable models for computational social systems. 

Index Terms—Automated machine learning (AutoML), cognitive diagnosis (CD), intelligent education, interpretable machine learning, multi-objective optimization, neural architecture search (NAS). 

# I. INTRODUCTION

W ITH the increasing integration of artificial intelligenceinto educational practices, the field is witnessing a sig- into educational practices, the field is witnessing a significant shift toward data-driven personalized learning [1], [2]. This transformation places intelligent education systems at the heart of modern computational social systems. A cornerstone of this transformation is the ability to accurately and transparently model a learner’s cognitive state. Cognitive diagnosis (CD), a class of latent trait models, has emerged as a critical technology for this purpose [3], [4]. By analyzing students’ response data (see Fig. 1), cognitive diagnosis models (CDMs) aim to infer their mastery over a set of fine-grained knowledge concepts. An accurate CDM provides actionable feedback for multiple stakeholders: it can guide learning pathways for students (the social actors) and offer diagnostic reports to help teachers refine their instructional strategies (the social system’s intervention) [5], [6]. However, the development of effective CDMs faces a fundamental tradeoff between predictive accuracy and model interpretability. On one hand, traditional models, such as the deterministic inputs, noisy “And” gate (DINA) model [7], [8], are built upon established psychometric principles. Their transparent and simple structures make the diagnostic results easy to understand, but their limited expressive power often fails to capture the complex, nonlinear relationships in student interaction data, thus constraining diagnostic accuracy. On the other hand, recent models leveraging deep learning, such as the neural cognitive diagnosis model (NCDM) [5], have demonstrated superior performance. Yet, their intricate architectures often operate as “opaque models,” obscuring the reasoning process behind the diagnosis. This lack of transparency can erode the trust of educators, who require a clear understanding of a model’s logic to confidently apply it in practice [9]. 

![](images/303c8ae4cb2726cfdd16c5d5fe7b706ac3ba51f3dd834021cf5d9e758234f12e.jpg)



Fig. 1. Illustration of CD process. Two students’ response records and Qmatrix are shown on the left, while the lower right gives the diagnosis results.


Furthermore, the design of these models has historically relied on a manual process, where interaction functions are handcrafted by experts. This manual approach is not only laborintensive but also inherently constrained by preexisting theoretical frameworks. It limits the exploration of a vast design space, potentially overlooking novel and more effective architectural patterns that lie beyond the scope of established theories [10]. This raises a critical research question: how can we automate the design of CDMs to simultaneously achieve high diagnostic performance and strong interpretability? 

To address this challenge, neural architecture search (NAS) [11], [12], a subfield of automated machine learning (AutoML), was utilized. NAS has achieved remarkable success in automatically discovering state-of-the-art neural network architectures for complex tasks in computer vision and natural language processing [13], [14]. Its principles are well-suited for automating and optimizing the design of CDMs, with promising applications already emerging in the broader educational domain [10], [15]. 

In this article, a novel framework named probabilistic modelbuilding for cognitive architecture search (PMB-CAS) was proposed. This framework formulates the design of a CDM’s core interaction function as a multi-objective optimization problem (MOOP). PMB-CAS seeks to identify a set of Pareto-optimal architectures that balance two competing objectives: diagnostic performance, measured by the area under the curve (AUC), and model interpretability, which is quantified using a new metric based on architectural simplicity. The core of PMB-CAS is an advanced search strategy based on estimation of distribution algorithms (EDAs) [16]. Instead of relying on conventional genetic operators such as random mutation, our approach iteratively trains a probabilistic model on a population of elite architectures. This model learns the structural properties of high-performing, interpretable designs, and new candidates are then sampled from the learned distribution. This enables a more guided and efficient exploration of a dedicated, tree-based search space tailored for CD. 

The major contributions of this work are summarized as follows. 

1) Framework Innovation: We propose PMB-CAS, the first multi-objective NAS (MONAS) framework specifically tailored for CD, which explicitly treats interpretability and accuracy as cooptimization objectives. 

2) Search Space Design: We design a novel tree-based search space that integrates both psychometric operators [e.g., item response theory (IRT) concepts] and deep neural operators, allowing for the discovery of hybrid models that bridge the gap between theory and data-driven methods. 

3) Interpretability Metric: We introduce an operatorweighted model interpretability score (MIS) that goes beyond simple depth constraints by penalizing semantic complexity, effectively guiding the search toward cognitively plausible architectures. 

The remainder of this article is organized as follows. Section II reviews related work in CD and NAS. Section III details the proposed PMB-CAS framework, including the search space, the multi-objective problem formulation, and the probabilistic model-guided search algorithm. Section IV presents the experimental setup and results. Finally, Section V concludes the article and discusses future work. 

# II. RELATED WORK

This section reviews the two primary research domains that ground our work. First, the evolution of CDMs shall be discussed, and then survey key paradigms in NAS, with a particular focus on its extension to multi-objective and interpretable optimization. 

# A. CDMs

The development of CDMs reflects a continuous effort to model the student learning process with greater fidelity. This field has evolved along two main trajectories. 

1) Psychometric Models: Early models are rooted in psychometrics and prioritize theoretical soundness. It is crucial to distinguish between two primary categories within this domain: continuous latent trait models and discrete CDMs. 

Foundational frameworks such as IRT [4], [17] represent the former. They model a student’s probability of a correct response as a function of a continuous latent ability (θ). While IRT serves as a robust baseline for predicting student performance (ranking), it is not inherently designed to infer the fine-grained, discrete mastery patterns of knowledge concepts defined by a Q-matrix. 

In contrast, CDMs are explicitly designed to diagnose discrete attribute mastery profiles. The DINA model [7] is a classic representative, employing a conjunctive logic where all required skills must be mastered. To address the rigidity of DINA, saturated models such as generalized DINA (G-DINA) [8] and probabilistic-attribute CDMs [18] were developed. G-DINA, in particular, covers all possible interaction effects among attributes and is often considered a theoretical upper bound for traditional psychometric models. However, these saturated models often require estimating exponential parameters (2K), which can lead to overfitting on sparse educational data. 

2) Deep Learning-Based Models: To enhance diagnostic accuracy, recent research has shifted toward leveraging deep learning. The NCDM was a seminal work in this direction, replacing the fixed interaction function of traditional models with a multilayer perceptron to learn more complex student-exercise relationships [5]. This innovation spurred a new line of research, with subsequent models incorporating more advanced neural components, such as factorization machines to model feature interactions [19], attention mechanisms for dynamic weighting of concepts [20], and graph neural networks to explicitly model the dependencies within the knowledge structure [21]. Despite their superior performance, these deep CDMs are still manually designed. This reliance on manual design creates a significant bottleneck, as it is not only labor-intensive but also constrains exploration to architectures that conform to human intuition, potentially overlooking novel and more effective designs. 

# B. NAS

NAS offers a paradigm to automate the discovery of optimal neural network architectures, mitigating the limitations of manual design [12]. A typical NAS framework comprises three core components: a search space defining the possible architectures, a search strategy to explore this space, and a performance estimation strategy to evaluate candidate architectures. Key search strategies include the following. 

1) Reinforcement Learning (RL): An RL agent is trained to sequentially generate architectural descriptions to maximize a reward signal, typically validation accuracy. While pioneering, this approach is known for its high computational cost [11]. 

2) Evolutionary Algorithms (EAs): Inspired by natural selection, a population of architectures is evolved over generations. EAs are highly parallelizable and robust against local optima. Genetic programming (GP), which evolves tree-structured programs, is a particularly relevant branch for our work, with significant advancements in its application to large-scale problems [22], [23], [24]. 

3) Differentiable Search: Methods such as DARTS relax the discrete search space into a continuous one, allowing for efficient gradient-based optimization [25]. However, these methods can suffer from high memory consumption and a performance gap between the searched architecture and the final evaluated one, an issue that subsequent works such as PC-DARTS have sought to address [26]. 

1) Multi-Objective and Interpretable NAS: While early NAS research focused on maximizing accuracy, practical applications often demand a balance between multiple, conflicting objectives. This led to the development of MONAS [27]. EAs are naturally suited for MONAS, as their population-based approach can efficiently approximate the Pareto front of nondominated solutions. A common application is finding architectures that balance accuracy with hardware efficiency metrics such as latency and model size [28]. 

More recently, aligning with the principles of Trustworthy AI, the scope of MONAS has expanded to include objectives such as fairness and interpretability [29]. For instance, some works have focused on discovering architectures that mitigate algorithmic bias [30]. Our work contributes to another critical frontier: treating model interpretability as a first-class, quantifiable objective. This is particularly vital in education, where opaque models can undermine user trust and hinder effective pedagogical intervention [31]. The formalization of a topologybased metric for interpretability enables the search to advance beyond simplistic proxies such as parameter count, facilitating a more meaningful, domain-aware search for models that are both accurate and structurally transparent. [10]. 

# C. The Intersection: NAS for AI in Education

The application of NAS in the educational domain is a nascent but promising research area. To date, efforts have primarily focused on the related task of knowledge tracing (KT), which models the temporal evolution of student knowledge [32]. For example, AutoDKT applied NAS to discover more effective recurrent cells for a DKT model [33]. While these studies validate the potential of NAS in education, the specific challenge of automating the design of static CDMs—particularly through a multi-objective lens that explicitly optimizes the accuracy-interpretability tradeoff—remains largely unexplored [34], [35]. This article aims to fill this critical research gap, presenting, to the best of our knowledge, the first probabilistic model-building MONAS framework designed specifically for CD. 

# III. THE PROPOSED PMB-CAS FRAMEWORK

To address the fundamental tradeoff between accuracy and interpretability in CD, this article introduces the PMB-CAS framework. This section details the technical architecture of the approach. PMB-CAS is designed to function as an automated optimization framework, systematically exploring a vast space of potential models to discover novel, high-performing, and structurally transparent cognitive diagnosis cells (CD cells). The section first outlines the holistic workflow of the framework, then delves into the three core components that enable this automated discovery: a flexible and domain-aware search space, a principled multi-objective problem formulation, and a sophisticated, model-guided search algorithm. 

# A. Overall Framework

The primary objective of PMB-CAS is to automate the discovery of the core interaction function within a general CDM, a component called the CD cell. To navigate the near-infinite design landscape, the framework operates as an iterative, modelbased evolutionary process, illustrated in Fig. 2. This process is engineered to learn from its successes and intelligently guide its subsequent exploration. 

The search cycle begins by initializing a diverse population of candidate architectures, each represented as a computation tree. In each generation, every candidate is instantiated as a complete CDM, trained for a limited number of epochs on a validation set, and evaluated to obtain its objective scores for accuracy and interpretability. Based on these two, often conflicting, objectives, an elite selection mechanism, guided by the well-established NSGA-II algorithm [36], identifies a subset of nondominated architectures—the current best solutions. The key step of our framework follows: these elite architectures are used to update a probabilistic model, which distills their collective structural wisdom by learning the statistical properties of their highperforming patterns. A new, potentially superior generation of architectures is then produced by sampling from this learned model, supplemented by a small fraction of randomly generated architectures to maintain population diversity and prevent premature convergence. This “evaluate-learn-sample” cycle continues for a predefined number of generations, progressively pushing the population toward the true Pareto-optimal front. 

![](images/79536777a8362dd0f759d7e9886551f2acb145dbe9b26cb25d6ef6d69c542a52.jpg)



Fig. 2. Overall flowchart of the proposed PMB-CAS framework. The search process iterates through evaluation, elite selection, probabilistic model updating, and guided sampling to discover a set of Pareto-optimal architectures.


# B. Search Space Formulation

A well-designed search space is critical for the success of any NAS algorithm. It must be expressive enough to contain novel, high-performance architectures, yet constrained enough to be searched efficiently. The search space of CD cell is formulated based on computational tree representation, where any candidate architecture ${ \mathcal { A } } \in { \mathcal { S } }$ is a tree. This representation is inherently compositional, allowing simple functions to be combined into complex interaction models, and its structure is naturally aligned with our goal of topological interpretability. The leaf nodes of the tree are input terminals, and the internal nodes are functional operators. 

1) Terminal Set (Leaf Nodes): The leaf nodes provide the fundamental inputs for the diagnostic process. Drawing from established CD theories, a terminal set consisting of four distinct, preprocessed feature embeddings was defined. This multifaceted input provides a rich foundation for modeling the nuanced interactions between a student and an exercise. 

a) Student proficiency $( \mathbf { h } _ { s } \in \mathbb { R } ^ { D _ { k } } ) \colon$ An embedding vector representing the latent proficiency of a student on each of the $D _ { k }$ knowledge concepts. 

b) Exercise difficulty $( \mathbf { h } _ { e , d i f f } \in \mathbb { R } ^ { D _ { k } } ) \colon$ An embedding vector representing the difficulty of an exercise with respect to each knowledge concept. 

c) KC relevance $( \mathbf { h } _ { k c } \in \mathbb { R } ^ { D _ { k } } ) \colon$ A vector, typically derived from the Q-matrix, indicating which knowledge concepts are measured by a given exercise. 

d) Exercise discrimination $( \mathbf { h } _ { e , d i s c } \in \mathbb { R } ^ { 1 } )$ : A scalar embedding representing the overall ability of an exercise to distinguish between students of different proficiency levels. 

This expanded set of inputs, particularly the inclusion of an explicit discrimination parameter, provides a richer basis for interaction modeling compared to prior works that relied on fewer inputs [10]. 

2) Function Set (Operator Nodes): The internal nodes are selected from a comprehensive set of operators O, which serve as the building blocks of the diagnostic logic. The set, detailed in Table I, includes a variety of unary, binary, and higher arity operators, ranging from basic arithmetic to learnable neural network layers. 

A key innovation in our search space is the inclusion of a high-arity, domain-specific operator, “NCDM-Core.” This operator encapsulates the core interaction function from the influential NCDM model [5], which is given by 

$$
f (h _ {s}, h _ {e, \text { diff }}, h _ {k c}, h _ {e, \text { disc }}) = h _ {e, \text { disc }} \cdot \sigma ((h _ {s} - h _ {e, \text { diff }}) \odot h _ {k c}) \tag {1}
$$

where $\sigma ( \cdot )$ denotes the Sigmoid activation function, and $\odot$ represents the element-wise (Hadamard) product. By incorporating this validated pattern as a single, powerful building block, the search algorithm was enabled to leverage existing domain knowledge. This hybrid approach allows our search to stand on the shoulders of giants, using established knowledge as a powerful starting point for further innovation rather than attempting to rediscover everything from scratch. 


TABLE I OPERATOR SET FOR THE CD CELL SEARCH SPACE


<table><tr><td>Operator</td><td>Arity</td><td>Expression</td><td>Output Dim.</td></tr><tr><td colspan="4">Unary Operators</td></tr><tr><td>Abs</td><td>1</td><td><eq>|x|</eq></td><td>Same as input</td></tr><tr><td>Inv</td><td>1</td><td><eq>1/(x + \epsilon)</eq></td><td>Same as input</td></tr><tr><td>Square</td><td>1</td><td><eq>x^{2}</eq></td><td>Same as input</td></tr><tr><td>Sqrt</td><td>1</td><td><eq>\text{sign}(x)\sqrt{|x|}</eq></td><td>Same as input</td></tr><tr><td>Tanh</td><td>1</td><td><eq>\text{tanh}(x)</eq></td><td>Same as input</td></tr><tr><td>Sigmoid</td><td>1</td><td><eq>\sigma(x)</eq></td><td>Same as input</td></tr><tr><td>Softplus</td><td>1</td><td><eq>\ln(1 + e^{x})</eq></td><td>Same as input</td></tr><tr><td>Sum</td><td>1</td><td><eq>\sum(x)</eq></td><td>Scalar (1)</td></tr><tr><td>Mean</td><td>1</td><td><eq>\text{mean}(x)</eq></td><td>Scalar (1)</td></tr><tr><td>FFN</td><td>1</td><td><eq>xW_{1} + b_{1}</eq></td><td>Scalar (1)</td></tr><tr><td>FFN_D</td><td>1</td><td><eq>xW_{2} + b_{2}</eq></td><td>Vector (<eq>D_{k}</eq>)</td></tr><tr><td colspan="4">Binary Operators</td></tr><tr><td>Add</td><td>2</td><td><eq>x + y</eq></td><td>Broadcast shape</td></tr><tr><td>Sub</td><td>2</td><td><eq>x - y</eq></td><td>Broadcast shape</td></tr><tr><td>Mul</td><td>2</td><td><eq>x \odot y</eq></td><td>Broadcast shape</td></tr><tr><td>Concat</td><td>2</td><td><eq>[x, y]W_{3} + b_{3}</eq></td><td>Vector (<eq>D_{k}</eq>)</td></tr><tr><td colspan="4">Domain-Specific Operator</td></tr><tr><td>NCDM-Core</td><td>4</td><td>Eq. 1</td><td>Vector (<eq>D_{k}</eq>)</td></tr></table>

# C. Multi-Objective Formulation

At the heart of our framework lies the formulation of CDM design as a MOOP. This work moves beyond the singular pursuit of accuracy to embrace the inherent tension between performance and interpretability. The goal is to find architectures that achieve a principled compromise between these two competing virtues. The objective vector $F ( A )$ for an architecture A is defined as 

$$
\max _ {\mathcal {A} \in \mathcal {S}} F (\mathcal {A}) = (f _ {\mathrm{auc}} (\mathcal {A}), f _ {\mathrm{int}} (\mathcal {A})). \tag {2}
$$

1) Objective 1: Diagnostic Performance (AUC) $( f _ { a u c } ) { : }$ The first objective measures the predictive performance of the model. The AUC on a held-out validation dataset $\mathcal { D } _ { \mathrm { v a l } }$ was used as the primary metric. AUC provides a robust and widely accepted measure of a model’s ability to discriminate between correct and incorrect responses 

$$
f _ {\text { auc }} (\mathcal {A}) = \text { AUC } (\text { model } (\mathcal {A}), \mathcal {D} _ {\text { val }}) \tag {3}
$$

where model(A) is the instantiated and trained CDM with the architecture A. 

2) Objective 2: Operator-Weighted Topological Interpretability $( f _ { i n t } ) \colon$ It is important to clarify that MIS serves as a topological proxy for interpretability rather than a comprehensive measure of cognitive semantics. By penalizing operatorweighted depth and node complexity, PMB-CAS identifies structural regularities that align with the parsimony principle in psychometrics. The cognitive intuition is that models with simpler, more compact structures are easier for humans to parse and trust [37]. While MIS provides a quantifiable objective to guide the search toward structural transparency, we acknowledge that it acts as a structural constraint to discourage high-complexity “opaque models” operations rather than a complete measure of semantic explainability. 


TABLE II PREDEFINED COMPLEXITY COST C(op) FOR EACH OPERATOR


<table><tr><td>Operator Category</td><td>Operators</td><td>Cost C(op)</td></tr><tr><td>Arithmetic</td><td>+, -, ×, /, Square, Abs</td><td>1</td></tr><tr><td>Activation</td><td>Sigmoid, Tanh, Softplus</td><td>2</td></tr><tr><td>Aggregation</td><td>Sum, Mean</td><td>2</td></tr><tr><td>Neural (Black-box)</td><td>FFN, FFN_D, Concat</td><td>5</td></tr><tr><td>Domain-Specific</td><td>NCDM_Core</td><td>5</td></tr></table>

Note: Bold entries indicate the optimal values or best performance for each respective metric. 

Standard topological metrics (e.g., node count) can be misleading, as they often treat all operators as equally complex. This is problematic when a search space includes both simple operators (e.g., add) and composite functions (e.g., NCDM-Core). To provide a fairer and more meaningful measure of complexity, we introduce a refined, operator-weighted interpretability score (MIS). We assign a complexity weight, $C ( o p )$ , to each operator based on its computational nature (e.g., $C ( \mathsf { a d d } ) = 1$ and $C ( \mathtt { N C D M - C o r e } ) = 5 )$ . Our proposed metric is then defined as a normalized score based on the weighted tree depth d and the total weighted complexity $C _ { \mathrm { t o t a l } }$ 

$$
C _ {\text { total }} (\mathcal {A}) = \sum_ {i \in \text { nodes } (\mathcal {A})} C (o p _ {i})
$$

$$
f _ {\text { int }} (\mathcal {A}) = \max \left(0, 1 - \left(w _ {d} \frac {d}{d _ {\max}} + w _ {c} \frac {C _ {\text { total }} (\mathcal {A})}{C _ {\max}}\right)\right) \tag {4}
$$

where $d _ { \mathrm { m a x } }$ and $C _ { \mathrm { m a x } }$ are normalization factors, and $w _ { d } , w _ { c }$ are weighting hyperparameters. This formulation explicitly penalizes deeper and more intrinsically complex computation trees, providing a robust measure of simplicity. The score is designed to be maximized (higher is better and more interpretable), with an upper bound of 1.0. 

To quantify the interpretability, we assign different cost values $C ( o p )$ to operators based on their cognitive transparency. As detailed in Table II, basic arithmetic operators (e.g., + and −) are assigned a low cost (C = 1) as they represent transparent logical steps. In contrast, “opaque model” neural components (e.g., FFN and Concat) and domain-specific modules (e.g., NCDM-Core) are penalized with a higher cost $( C = 5 )$ to discourage the search algorithm from over-relying on complex, uninterpretable transformations unless they significantly improve accuracy. 

Given these two conflicting objectives, this work seeks not a single “best” model, but the Pareto front: a set of solutions where each is a champion in its own right. An architecture $\mathcal { A } ^ { \ast }$ is on the Pareto front if no other architecture $\mathcal { A } ^ { \prime }$ exists that is strictly better on one objective without being worse on the other. As illustrated in Fig. 3, the resulting front provides decisionmakers with a diverse portfolio of high-quality models, enabling a principled tradeoff between accuracy and interpretability. 

# D. Probabilistic Model-Guided Search Algorithm

To efficiently solve the defined MOOP, this work moves beyond conventional evolutionary methods that rely on stochastic, problem-agnostic genetic operators such as mutation and crossover. While powerful in general applications, such operators represent a form of undirected stochastic variation in the context of architecture search. They operate without any memory or accumulated knowledge of the search space, often disrupting well-performing structural motifs or “building blocks” just as frequently as they produce improvements. This undirected exploration can be highly inefficient when navigating a complex and structured search space like that of CDMs. 

![](images/d3b3704e569dd24a43d6cfb1a3618997f40d696b65b6fe2d72a19896edee8431.jpg)



Fig. 3. Visualization of the discovered Pareto-optimal architectures and their structures. (a) Tradeoff between validation AUC and the topology-based interpretability score. (b)–(e) Computation graphs of four representative architectures, highlighting the diversity from (a) simple structures to (d) novel compositions.


Instead, this framework, PMB-CAS, employs a more intelligent search strategy rooted in the EDA paradigm [16]. The core idea of EDA is to replace blind variation with explicit, modelguided generation. The search process iterates through a selectmodel-sample cycle. Specifically, at each generation, a set of elite architectures is selected from the current population based on their nondominated rank on the Pareto front. Then, rather than applying crossover or mutation, we construct a probabilistic model—in our case, a model capturing the topological properties and component choices of these elite solutions. This model essentially learns a statistical representation of what makes a high-performing architecture. New candidate architectures are then generated by sampling from this learned model, ensuring that the subsequent generation is inherently biased toward the promising, high-performing regions of the search space. 

This model-based approach is conceptually similar to surrogate-assisted evolutionary computation (SAEC) [38], as both leverage models to accelerate the search. However, a key distinction is that while SAEC typically builds a surrogate to approximate the expensive fitness function, our PMB-CAS approach models the distribution of elite solutions themselves. This allows us to directly generate novel, high-potential architectures, rather than merely predicting the performance of randomly generated ones. This process of explicitly learning 

# Algorithm 1: The PMB-CAS Algorithm.

1: Initialize population $\mathcal { P } _ { 0 }$ with N random architectures. 

2: Initialize probabilistic model ${ \mathcal { M } } _ { p r o b } .$ 

3: for $t = 0 , 1 , \ldots , T _ { m a x } - 1$ do 

4: Evaluate all $\mathcal { A } \in \mathcal { P } _ { t }$ to get their objective vectors $F ( A )$ . 

5: Select elite set $\mathcal { P } _ { e l i t e }$ from $\mathcal { P } _ { t }$ using non-dominated sorting and crowding distance [36]. 

6: Update the probabilistic model $\mathcal { M } _ { p r o b }$ by learning from the structural statistics of architectures in $\mathcal { P } _ { e l i t e }$ . 

7: Initialize the next population $\mathcal { P } _ { t + 1 } = \varnothing .$ 

8: Sample $N _ { s a m p l e }$ new architectures from $\mathcal { M } _ { p r o b }$ and add to $\mathscr { P } _ { t + 1 }$ . 

9: Add $N _ { r a n d o m }$ new random architectures to $\mathcal { P } _ { t + 1 }$ to maintain diversity. 

10: $\mathcal { P } _ { t + 1 }$ becomes the population for the next generation. 

11: end for 

12: return Final non-dominated set from all evaluated architectures. 

and exploiting the problem’s structural regularities, as will be detailed in Section I, is fundamentally more suited for discovering sophisticated and well-formed CDMs. 

Computational Complexity Analysis: The theoretical complexity of PMB-CAS per generation is $O ( N \cdot ( { \mathcal { C } } _ { \mathrm { e v a l } } + { \mathcal { C } } _ { \mathrm { u p d a t e } } + { \mathcal { C } } _ { \mathrm { s a m p l e } } ) )$ , where N is the population size. The most significant computational bottleneck is $\mathcal { C } _ { \mathrm { e v a l } } ,$ , which involves training N individual CDMs. The complexity of updating the probabilistic model $( \mathcal { C } _ { \mathrm { u p d a t e } } )$ and sampling new architectures $( { \mathcal { C } } _ { \mathrm { s a m p l e } } )$ scales linearly with the population size and the maximum tree depth $d _ { \operatorname* { m a x } } ,$ i.e., $O ( N \cdot d _ { \operatorname* { m a x } } )$ . Compared to RL-based NAS, this model-building approach is more sample-efficient as it explicitly captures the structural regularities of elite solutions. 

1) Variable-Depth Tree Generation: A key technical aspect of our EDA is its ability to generate variable-depth trees. Our probabilistic model (detailed below) learns the conditional probabilities of transitioning from a parent operator to either another operator or a leaf terminal, based on the context (e.g., parent type, child position, and current depth). New architectures are generated via a recursive sampling process: 

a) We first sample a root operator from the model’s root distribution. 

b) Then, for each of the operator’s arguments, we sample its node type (operator versus leaf) from the learned probability P (Type|Parent, Pos, Depth). 

c) If the type is “leaf,” we sample a specific leaf $( \mathrm { e } . \mathrm { g } . , h _ { s } )$ and the branch terminates. 

d) If the type is “operator,” we sample a specific operator (e.g., “add”), and this recursive process is repeated for the new operator’s children. 

This recursive generation naturally produces trees of varying depths and topologies, guided by the learned statistics of elite solutions rather than predefining a fixed structure. The initial population $( \mathcal { P } _ { 0 } )$ is generated using a standard ramped half-andhalf method to ensure diverse starting structures. 

2) Probabilistic Model Construction: After identifying the elite set $\mathcal { P } _ { \mathrm { e l i t e } } .$ , a probabilistic model $\mathcal { M } _ { \mathrm { p r o b } }$ was constructed that captures their structural regularities. The model is a collection of conditional probability distributions learned by parsing the elite trees and counting frequencies. This allows the model to capture the hidden syntax of effective CDMs. Key distributions include the following. 

a) Node production rule probability: For each parent operator $\mathrm { o p } _ { p }$ , the probabilistic model learns the probability of it producing a specific child node type (e.g., another operator and a leaf terminal). For instance, the model learns whether an “add” operation is more likely to take two leaf nodes as input or another nested operation. 

b) Operator/terminal selection probability: Given a parent operator and the designated type of a child node, the probabilistic model learns the probability of selecting a specific operator or terminal. For example, the model might learn that if the parent is a “sub” operation, its child is highly likely to be the “square” operator, thus capturing the valuable “square(sub(. . .))” motif discovered in our experiments. 

To prevent zero probabilities for unseen structures and encourage exploration, the model applies Laplace smoothing. The probability of selecting an operator $\mathrm { o p } _ { i }$ given a parent context c is as follows: 

$$
P (\mathrm{op} _ {i} | c) = \frac {\operatorname{count} (\mathrm{op} _ {i} , c) + \alpha}{\sum_ {j \in \mathcal {O}} (\operatorname{count} (\mathrm{op} _ {j} , c)) + | \mathcal {O} | \alpha} \tag {5}
$$

where α is the smoothing parameter. 

3) Temperature-Controlled Architecture Sampling: New architectures are generated by recursively sampling from the 


TABLE III STATISTICS OF THE DATASETS USED IN THE EXPERIMENTS


<table><tr><td>Statistic</td><td>ASSIST09</td><td>Junyi</td><td>SLP</td></tr><tr><td>No. of Students</td><td>4163</td><td>10747</td><td>2336</td></tr><tr><td>No. of Exercises</td><td>17746</td><td>708</td><td>3021</td></tr><tr><td>No. of KCs</td><td>123</td><td>25</td><td>749</td></tr><tr><td>No. of Logs</td><td>346924</td><td>707842</td><td>995680</td></tr><tr><td>Sparsity</td><td>99.53%</td><td>99.07%</td><td>99.86%</td></tr></table>

learned model $\mathcal { M } _ { p r o b }$ . To strategically balance exploration and exploitation, a temperature parameter $\tau \in ( 0 , 1 ]$ was introduced into the sampling process. The probability $p _ { i }$ of selecting an item i is adjusted using a softmax-like function 

$$
p _ {i} ^ {\prime} = \frac {p _ {i} ^ {1 / \tau}}{\sum_ {j} p _ {j} ^ {1 / \tau}}. \tag {6}
$$

This temperature acts as a focusing mechanism. When τ is high (e.g., 1.0), it encourages a broad global exploration, where the adjusted distribution $\quad { \overline { { p ^ { \prime } } } }$ is smooth, allowing the sampler to explore diverse and unconventional structures. As the search progresses, τ is gradually annealed toward 0. This “cooling” process sharpens the distribution, causing the sampler to increasingly exploit the high-probability patterns discovered in the elite solutions, facilitating fine-grained local exploitation. This strategic annealing allows PMB-CAS to first broadly explore the design space and then meticulously fine-tune the most promising architectural motifs. 

# IV. EXPERIMENTS

In this section, a series of experiments was conducted to comprehensively evaluate the effectiveness of our proposed PMB-CAS framework. Our evaluation is designed to answer the following research questions. 

1) RQ1: Do the architectures discovered by PMB-CAS outperform state-of-the-art, manually designed CDMs in diagnostic performance (as measured by AUC)? 

2) RQ2: What are the structural characteristics of the discovered architectures, and how do they represent a meaningful tradeoff between performance and interpretability? 

3) RQ3: How effective is the probabilistic model-building search strategy compared to standard search baselines in terms of search efficiency and stability? 

4) RQ4: Can the models discovered by PMB-CAS provide actionable diagnostic insights for educational practitioners? 

# A. Experimental Setup

1) Datasets: To evaluate the method on three widely used, real-world benchmark datasets for CD. Key statistics are provided in Table III. 

a) ASSISTments2009 (ASSIST09): A large-scale dataset from the ASSISTments intelligent tutoring system, containing student-exercise interaction logs with skill annotations [39]. 


TABLE IV PERFORMANCE COMPARISON ON THE ASSISTMENTS2009 DATASET


<table><tr><td rowspan="2">Method</td><td colspan="3">Accuracy Metrics (70%/30%)</td><td colspan="2">Efficiency &amp; Interpretability</td></tr><tr><td>ACC</td><td>RMSE</td><td>AUC</td><td>Topology Score</td><td>Param Complexity</td></tr><tr><td colspan="6">Baseline Predictive Models</td></tr><tr><td>IRT</td><td>0.6978</td><td>0.4458</td><td>0.7123</td><td>N/A</td><td>Low</td></tr><tr><td colspan="6">Psychometric CDMs</td></tr><tr><td>DINA [7]</td><td>0.6824</td><td>0.4805</td><td>0.7105</td><td>0.7208</td><td>Low</td></tr><tr><td>G-DINA [8]</td><td>0.6755</td><td>0.4888</td><td>0.7042</td><td>0.6500</td><td>High (<eq>2^K</eq>)</td></tr><tr><td colspan="6">Manually-Designed Deep CDMs</td></tr><tr><td>NCDM [5]</td><td>0.7305</td><td>0.4302</td><td>0.7421</td><td>0.4500</td><td>High</td></tr><tr><td>MCD [41]</td><td>0.7179</td><td>0.4348</td><td>0.7448</td><td>0.6158</td><td>Very High</td></tr><tr><td>KSCD [42]</td><td>0.7256</td><td>0.4320</td><td>0.7433</td><td>0.5500</td><td>High</td></tr><tr><td colspan="6">Discovered by PMB-CAS (Ours)</td></tr><tr><td>PMB-CAS-A (Interpretable)</td><td>0.7021</td><td>0.4512</td><td>0.7315</td><td>0.9200</td><td>Very Low</td></tr><tr><td>PMB-CAS-D (Accurate)</td><td>0.7432</td><td>0.4205</td><td>0.7752</td><td>0.8100</td><td>Low</td></tr></table>


Note: Bold entries indicate the optimal values or best performance for each respective metric. 



TABLE V PERFORMANCE COMPARISON ON THE JUNYI ACADEMY DATASET


<table><tr><td rowspan="2">Method</td><td colspan="3">Accuracy Metrics (70%/30%)</td><td colspan="2">Efficiency &amp; Interpretability</td></tr><tr><td>ACC</td><td>RMSE</td><td>AUC</td><td>Topology Score</td><td>Param Complexity</td></tr><tr><td colspan="6">Baseline Predictive Models</td></tr><tr><td>IRT</td><td>0.7856</td><td>0.3845</td><td>0.8123</td><td>N/A</td><td>Low</td></tr><tr><td colspan="6">Psychometric CDMs</td></tr><tr><td>DINA</td><td>0.7712</td><td>0.3956</td><td>0.8015</td><td>0.7208</td><td>Low</td></tr><tr><td>G-DINA</td><td>0.7688</td><td>0.3982</td><td>0.7955</td><td>0.6500</td><td>High</td></tr><tr><td colspan="6">Manually-Designed Deep CDMs</td></tr><tr><td>NCDM</td><td>0.8105</td><td>0.3654</td><td>0.8421</td><td>0.4500</td><td>High</td></tr><tr><td>MCD</td><td>0.8055</td><td>0.3712</td><td>0.8388</td><td>0.6158</td><td>Very High</td></tr><tr><td>KSCD</td><td>0.8089</td><td>0.3688</td><td>0.8405</td><td>0.5500</td><td>High</td></tr><tr><td colspan="6">Discovered by PMB-CAS (Ours)</td></tr><tr><td>PMB-CAS-A (Interpretable)</td><td>0.7956</td><td>0.3756</td><td>0.8256</td><td>0.9200</td><td>Very Low</td></tr><tr><td>PMB-CAS-D (Accurate)</td><td>0.8215</td><td>0.3544</td><td>0.8567</td><td>0.8100</td><td>Low</td></tr></table>


Note: Bold entries indicate the optimal values or best performance for each respective metric. 


b) Junyi: A large dataset from the Junyi Academy e-learning platform, focusing on mathematics exercises [40]. 

c) SLP: A dataset from a real-world online learning platform, containing a large number of knowledge concepts. For each dataset, each student’s records are chronologically split. During the NAS phase, 10% of the training data is held out as a validation set. For the final evaluation, models are retrained from scratch on the full training set and evaluated on the test set. 

The PMB-CAS was compared against several categories of baselines: 1) predictive models: IRT [4]; 2) psychometric CDMs: DINA [7] and G-DINA [8]; 3) deep CDMs: NCDM [5], MCD [41], KSCD [42], RNCD [43], and KANCD [44]; and 4) NAS baselines: RS and EA-NAS. 

2) Implementation Details: Our framework is implemented in PyTorch. The experiments were conducted on a mobile computing platform equipped with an NVIDIA GeForce RTX 4060 Laptop GPU and an Intel Core i7-13700H CPU. For the largest dataset (Assistments09), the complete search process for 50 generations required approximately 144 GPU hours. While the offline search phase is computationally intensive, the resulting CD cell is highly efficient for online deployment. The inference latency for a single student–item interaction is less than 2.5 ms, making it suitable for real-time educational applications. 

# B. Performance Comparison and Architecture Analysis (RQ1 & RQ2)

As shown in Tables IV–VI, the discovered PMB-CAS-D consistently outperforms state-of-the-art CDMs (RQ1). This superiority stems from its ability to discover novel interaction functions, such as the square(sub(·)) structure visualized in Fig. 4, which balances accuracy with a topology-based interpretability score (RQ2). 

To assess generalization, we observed that the core topological patterns discovered on ASSIST09 remain consistent when applied to other datasets, suggesting that PMB-CAS identifies intrinsic cognitive regularities rather than overfitting to a specific validation set. 


TABLE VI PERFORMANCE COMPARISON ON THE SLP DATASET


<table><tr><td rowspan="2">Method</td><td colspan="3">Accuracy Metrics (70%/30%)</td><td colspan="2">Efficiency &amp; Interpretability</td></tr><tr><td>ACC</td><td>RMSE</td><td>AUC</td><td>Topology Score</td><td>Param Complexity</td></tr><tr><td colspan="6">Baseline Predictive Models</td></tr><tr><td>IRT</td><td>0.6545</td><td>0.4856</td><td>0.6856</td><td>N/A</td><td>Low</td></tr><tr><td colspan="6">Psychometric CDMs</td></tr><tr><td>DINA</td><td>0.6423</td><td>0.4912</td><td>0.6712</td><td>0.7208</td><td>Low</td></tr><tr><td>G-DINA</td><td>0.6455</td><td>0.4895</td><td>0.6755</td><td>0.6500</td><td>High</td></tr><tr><td colspan="6">Manually-Designed Deep CDMs</td></tr><tr><td>NCDM</td><td>0.7012</td><td>0.4556</td><td>0.7256</td><td>0.4500</td><td>High</td></tr><tr><td>MCD</td><td>0.6956</td><td>0.4612</td><td>0.7188</td><td>0.6158</td><td>Very High</td></tr><tr><td>KSCD</td><td>0.7001</td><td>0.4589</td><td>0.7223</td><td>0.5500</td><td>High</td></tr><tr><td colspan="6">Discovered by PMB-CAS (Ours)</td></tr><tr><td>PMB-CAS-A (Interpretable)</td><td>0.6856</td><td>0.4656</td><td>0.7012</td><td>0.9200</td><td>Very Low</td></tr><tr><td>PMB-CAS-D (Accurate)</td><td>0.7156</td><td>0.4412</td><td>0.7512</td><td>0.8100</td><td>Low</td></tr></table>


Note: Bold entries indicate the optimal values or best performance for each respective metric. 



Final Model Performance Comparison on Test Sets (AUC)


![](images/2b7e3292bb5c57105b37ee5e5495f610cf5d325d32ecd780bbb9a4f97bd15b4f.jpg)



Fig. 4. Final model performance comparison on the 70%/30% test sets. This chart visualizes the AUC for our best-performing discovered model (PMB-CAS-D) against key traditional and deep learning baselines across all three datasets. PMB-CAS-D consistently achieves state-of-the-art or highly competitive performance.


Understanding FLOPs (A Tale of Two Paths): The vast difference in FLOPs is a direct result of a critical branching logic in our framework. 

1) Architectures whose core “CD cell” outputs a vector (such as A, B, and C) must use a heavy three-layer 

MLP for prediction, resulting in a high-FLOPs path (393k FLOPs). 

2) In contrast, architectures such as PMB-CAS-D, which produce a scalar latent factor, bypass this MLP, leading to drastically lower FLOPs (512). This shows the framework can find both accurate and computationally efficient models. 

![](images/b99c5354b924b9869f2a8552bd641573c545980d14f651f95965df902e81345c.jpg)



Fig. 5. Comparison of search efficiency on ASSIST09. The y-axis represents the best validation AUC found so far. PMB-CAS consistently finds better architectures faster than the EA-NAS and random search baselines.


![](images/9d5078e9ad96eba4504b67216ad3e8b05b20ada23d1449d87e14fb9ee250e9ff.jpg)



Fig. 6. Distribution of final best validation AUC for different search strategies over multiple runs. The box plot shows that PMB-CAS (left) not only achieves a higher median performance but is also more stable, consistently finding superior solutions compared to EA-NAS and random search.


# C. Ablation Study on Search Strategy (RQ3)

To isolate the contribution of our probabilistic modelbuilding strategy, PMB-CAS was compared against EA-NAS and RS on the ASSIST09 dataset. The comparison provides a two-fold confirmation of our method’s superiority. 

First, Fig. 5 plots the highest validation AUC found by each method over 50 generations. The results clearly show that PMB-CAS demonstrates significantly higher search efficiency, discovering high-performing architectures much earlier than both baselines. Second, to assess the stability and reliability of the final results, the distribution of the best AUC achieved over multiple independent runs in Fig. 6 was compared. The box plot reveals that PMB-CAS not only finds solutions with a higher median AUC but also exhibits a much smaller variance compared to EA-NAS and especially Random Search. This indicates that our method is more robust and consistently converges to high-quality solutions. 

![](images/fe88db0601e102726ed214a86d8b656f2edc791d4771326394b1245f83fbaa2e.jpg)


![](images/a2c6aec7fd95fe667855294e7f7d257da3f17921401f3135b97fbac064dfa980.jpg)



Fig. 7. Examples of diagnostic visualizations generated from a discovered model (PMB-CAS-C). (a) Proficiency heatmap allows teachers to compare knowledge states across multiple students. (b) Detailed proficiency radar chart provides a multidimensional “knowledge snapshot” for a single student, enabling targeted interventions. (a) Proficiency heatmap for group analysis. (b) Individual knowledge state radar chart.


Together, these results decisively answer RQ3, confirming that learning a probability distribution of elite architectures enables a more intelligent, efficient, and stable search process. 

# D. Case Study on Diagnostic Insights (RQ4)

To answer RQ4 and demonstrate practical utility, the trained PMB-CAS-C model was used to generate diagnostic reports. As shown in Fig. 7, the model’s output can be transformed into actionable feedback. The proficiency heatmap [Fig. 7(a)] allows a teacher to quickly compare the knowledge states of multiple students, identifying common areas of difficulty within a group. The proficiency radar chart [Fig. 7(b)] provides a detailed, multidimensional “knowledge snapshot” for an individual student. These visualizations bridge the gap between advanced ML and practical application, affirming the utility of our approach for personalized learning. 

# E. Parameter Sensitivity Analysis

To investigate the robustness of our proposed MIS, we conducted a sensitivity analysis on the hyperparameters $w _ { d }$ (depth weight) and $w _ { c }$ (complexity weight). As illustrated in Fig. 8, we varied both weights from 0.1 to 1.0 and calculated the Kendall’s τ rank correlation between the architecture rankings produced by the new weights and our default setting $( w _ { d } = 0 . 5 , w _ { c } =$ 0.5). The heatmap shows a consistently high correlation (>0.85) across a wide central region, indicating that the preference of PMB-CAS for interpretable architectures is stable and not sensitive to minor hyperparameter perturbations. 

![](images/2d402fcc65883b355cd13eafaf269bc9f6f8e8bd42a7ba3417b0b59f74d70a35.jpg)



Fig. 8. Sensitivity analysis of the interpretability score (MIS). The high Kendall’s τ correlation indicates that the ranking of architectures is robust to variations in weighting parameters.


# V. CONCLUSION AND FUTURE WORK

In this article, we introduced PMB-CAS, a novel framework for automating the design of CDMs, a critical component in modern educational computational social systems. By formulating the design task as a biobjective optimization problem, our approach addresses a core challenge in the field: balancing diagnostic accuracy with a refined, operator-weighted measure of topological interpretability. This addresses a key flaw in prior metrics and provides a fairer assessment of model complexity. The search process is guided by a probabilistic model that learns from elite architectures, enabling a more efficient and intelligent exploration of a vast, tree-based search space. 

The extensive experiments on three benchmark datasets provide compelling evidence for the effectiveness of PMB-CAS. The discovered architectures not only significantly outperform a wide range of manually designed CDMs but also remain highly competitive with the latest state-of-the-art models. The ablation studies further confirm that our probabilistic model-guided search strategy is markedly more efficient than standard evolutionary or random search baselines. Crucially, our framework produces a Pareto front of solutions, offering practitioners and stakeholders in social systems a valuable spectrum of models that tradeoff between high performance and structural transparency. The case studies demonstrate that these models can generate actionable, fine-grained diagnostic reports, highlighting their practical utility in real-world educational scenarios. 

While PMB-CAS demonstrates promising results, it has limitations. First, the current search space, though hybrid, is still limited to predefined operators and may not capture high-order complex interactions found in some specific learning domains. Second, the interpretability is primarily topological; connecting these discovered structures to deeper pedagogical theories requires further expert validation. 

Future work will focus on two directions: 1) incorporating “expert-in-the-loop” mechanisms where educational experts can inject constraints into the NAS process; and 2) extending the framework to handle dynamic CD tasks where student knowledge states evolve over time. 

# REFERENCES



[1] L. Chen, P. Chen, and Z. Lin, “Artificial intelligence in education: A review,” IEEE Access, vol. 8, pp. 75264–75278, 2020. 





[2] R. S. Baker, “Data mining for education,” in International Encyclopedia of Education. Amsterdam, The Netherlands: Elsevier, 2010, pp. 112– 118. 





[3] Y. Gao, Q. Liu, E. Chen, H. Liu, and S. Wang, “A survey on cognitive diagnosis: Theories and applications,” ACM Comput. Surv. (CSUR), vol. 54, no. 9, pp. 1–37, 2021. 





[4] F. M. Lord, Applications of Item Response Theory to Practical Testing Problems. New York, NY, USA: Routledge, 1980. 





[5] F. Wang et al., “Neural cognitive diagnosis model for intelligent education systems,” in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 01, 2020, pp. 259–266. 





[6] T. Zhu, P. Wang, T. Li, R. Wang, and F.-Y. Wang, “Personalized educational systems based on computational intelligence,” IEEE Trans. Computat. Social Syst., vol. 8, no. 4, pp. 847–857, Aug. 2021. 





[7] J. De La Torre, “DINA model and parameter estimation: A didactic,” J. Educ. Behav. Statist., vol. 34, no. 1, pp. 115–130, 2009. 





[8] J. De la Torre, “The generalized DINA model framework,” Psychometrika, vol. 76, pp. 179–199, 2011. 





[9] F. Wang, Q. Liu, E. Chen, Z. Huang, Y. Yin, and S. Wang, “Interpretable cognitive diagnosis with neural networks,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 4, pp. 3542–3555, Apr. 2021. 





[10] S. Yang et al., “An evolutionary multiobjective neural architecture search approach to advancing cognitive diagnosis in intelligent education,” IEEE Trans. Evol. Comput., vol. 29, no. 6, pp. 2431–2445, Dec. 2025, doi: 10.1109/TEVC.2024.3392424. 





[11] B. Zoph and Q. V. Le, “Neural architecture search with reinforcement learning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017. 





[12] T. Elsken, J. H. Metzen, and F. Hutter, “Neural architecture search: A survey,” J. Mach. Learn. Res., vol. 20, no. 55, pp. 1–21, 2019. 





[13] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. Int. Conf. Mach. Learn., PMLR, 2019, pp. 6105–6114. 





[14] C. Liu et al., “Progressive neural architecture search,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 19–34. 





[15] S. Wang, Q. Liu, E. Chen, Z. Huang, F. Wang, and Y. Gao, “Auto-SC: An automated framework for student characteristic analysis with neural architecture search,” in Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining, 2021, pp. 1769–1779. 





[16] P. Larrañaga and J. A. Lozano, Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation, vol. 2. Cham, Switzerland: Springer Sci. & Bus. Media, 2001. 





[17] S. E. Embretson and S. P. Reise, Item Response Theory for Psychologists. New York, NY, USA: Routledge, 2013. 





[18] J. Chen, J. de la Torre, and Z. Zhang, “A cognitive diagnosis model for probabilistic-attribute hierarchies,” Front. Psychol., vol. 9, p. 997, 2018. 





[19] Y. Cheng et al., “Neural factorization for cognitive assessment,” in Proc. 28th ACM Int. Conf. Inf. Knowl. Manage., 2019, pp. 2277–2280. 





[20] Q. Liu et al., “EKT: Exercise-aware knowledge tracing for student performance prediction,” in Proc. IEEE Int. Conf. Data Mining (ICDM), Piscataway, NJ, USA: IEEE, 2019, pp. 448–457. 





[21] C. Gao, J. Huang, S. Lan, J. Zhang, and J. You, “Graph-based knowledge tracing: A survey,” in Proc. 30th Int. Joint Conf. Artif. Intell., 2021, pp. 4362–4369. 





[22] E. Real et al., “Large-scale evolution of image classifiers,” in Proc. Int. Conf. Mach. Learn., PMLR, 2017, pp. 2902–2911. 





[23] J. R. Koza, Genetic Programming: On the Programming of Computers by Means of Natural Selection. Cambridge, MA, USA: MIT Press, 1992. 





[24] F. Policistico, G. S. G. de, F. C. Junior, S. R. de, and L. Meira, “A survey on large-scale genetic programming,” IEEE Trans. Evol. Comput., vol. 25, no. 5, pp. 829–848, Oct. 2021. 





[25] H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable architecture search,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019, pp. 1–13. 





[26] Y. Xu et al., “PC-DARTS: Partial channel connections for memoryefficient architecture search,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020, pp. 1–13. 





[27] A. A. A. Abd Allateef, T. T-Abd-El-Hafeez, and M. M. Fouad, “A survey on multi-objective neural architecture search,” Artif. Intell. Rev., vol. 55, no. 5, pp. 3777–3828, 2022. 





[28] Z. Lu et al., “NSGANetV2: Evolutionary multi-objective surrogateassisted neural architecture search,” in Proc. Eur. Conf. Comput. Vis., Cham, Switzerland: Springer, 2020, pp. 426–441. 





[29] L. Floridi et al., “Establishing the rules for building trustworthy AI,” Nature Mach. Intell., vol. 1, no. 6, pp. 261–262, 2019. 





[30] R. Gong et al., “AlphaD3M: An open-source library for automated machine learning and model selection,” in Proc. Automat. Mach. Learn., Cham, Switzerland: Springer, 2021, pp. 187–200. 





[31] W. Wang et al., “Can we automate scientific reviewing? A position paper,” in Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining, 2021, pp. 3780–3789. 





[32] C. Piech et al., “Deep knowledge tracing,” in Proc. Adv. Neural Inf. Process. Syst., vol. 28, 2015, pp. 505–513. 





[33] S. Wang, Q. Liu, E. Chen, Z. Huang, F. Wang, and Y. Su, “AutoDKT: An automated framework for dynamic knowledge tracing with neural architecture search,” in Proc. Int. Conf. Manage. Data, 2023, pp. 1–11. 





[34] Y. Luo et al., “Interpretable student performance prediction in online learning,” IEEE Trans. Computat. Social Syst., vol. 10, no. 2, pp. 789– 801, Apr. 2023. 





[35] B. Hu, Z. Wang, and H. Liu, “A review of affective computing in education,” IEEE Trans. Computat. Social Syst., vol. 7, no. 2, pp. 343– 358, Apr. 2020. 





[36] K. Deb, A. Pratap, S. Agarwal, and T. A. M. T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, Apr. 2002. 





[37] L. Hancox-Li, “A systematic review of interpretable machine learning,” in Proc. Conf. Fairness, Accountability, Transparency, 2020, pp. 609– 609. 





[38] Y. Jin, “Surrogate-assisted evolutionary computation: A review,” IEEE Trans. Evol. Comput., vol. 15, no. 5, pp. 601–617, Oct. 2011. 





[39] M. Feng, N. T. Heffernan, and K. R. Koedinger, “Addressing the assessment challenge with an online system that tutors as it assesses,” J. Educ. Data Mining, vol. 1, no. 1, pp. 31–66, 2009. 





[40] Y.-J. Chang et al., “Junyi academy: A large-scale online learning platform and its research opportunities,” in Proc. 2nd Workshop Data-Driven Educ., 2015, pp. 1–6. 





[41] T. He et al., “Multi-component learning for cognitive diagnosis,” in Proc. 31st ACM Int. Conf. Inf. Knowl. Manage., 2022, pp. 740–749. 





[42] F. Long et al., “KSCD: A knowledge-structure-aware causal-based diagnosis framework for student performance prediction,” in Proc. IEEE Int. Conf. Data Mining (ICDM), Piscataway, NJ, USA: IEEE, 2021, pp. 369–378. 





[43] W. Gao et al., “RCD: Relation-aware cognitive diagnosis for intelligent education systems,” in Proc. 44th Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2021, pp. 1533–1542. 





[44] J. Shen, Y. Liu, Q. Liu, E. Chen, Z. Huang, and S. Wang, “Assessing student’s dynamic knowledge state with knowledge-aware attentive network,” in Proc. 30th ACM Int. Conf. Inf. Knowl. Manage., 2021, pp. 1619–1628. 

