# MODULUS Library, K. Shukla

### A quick Intro in GPs

Think: how do you sample from a normal distribution? 
* Metropolis hastling algorithm

What is covariance? A correlation coefficient. Why are covarience in GP symmetric positive? 
"Correlation coefficient is nothing but a distance". 

Why Gaussian Processes fell behind of neural networks? Because of the size of the covariance matrix. 

### Can there be non-positive (symmetric) correlation matrices? 
* Negative definite: no it cannot, because it measures distance. 

Why? because of the definition of the positive-definie definition and the construction of the correlation matrix. 

A bit of the notes he said:
* So one samples **x**~_N(μ, σ)_
* Where **x**$=[x_1,X_2]$ with both being non correlated.

But aaaah we're computing a correlation kernel... 

oxi, I need to learn better this concept. 

BLUE: best linear unbiased estimator

**THERE IS A CODE THAT TEACHS HOW TO SAMPLE FROM A GAUSSIAN PROCESS!**

No correlation = high variance. 

Take a NN, just one layer, with infinity neurons ---> that is a gaussian process. 

* If there were specialized processors units that would focus on computing the inverse of a matrix, there would still be an issue with the inverse of the kernel. "It's not that easy". 
    * I was thinking that, if there were like 'GPUs' but for inverse operations, then GPs would overcome NNs. No? 
* SURROGATES FOR THE INVERSE OF THE MATRICES: that's what he understood of my question.
    * This is why he said "it's not that easy"; I think he thought I wanted a Onet for the inverse of a matrix. ohhhhh. 
    * The inverse of a derivative, I think 

* BLAS (Basic Linear Algebra Subprograms):: seems to be a programming language. This is in every computer. If you go to the source code of torch, for instance, it's written in BLAS. BLAS has thousands of lines to compute a the inverse of a matrix; this is mainly because of the architecture of CPUs. 


## HOW MUCH ARE HUMANS LIMITATED BY THEIR SENSES / TECHONOLOGY?

So, we cannot see wavelengths that are outise $\lambda = [400,600]$nm, so, that's a limitation. Our brain receives thousands of information inputs, but still, can only process a fraction of it. 

Nowadays, we have techniligy, we have AI, like cgatGPT wich depends on trillions of parameters; still, that's limited. 

GPs are a extremelly powerful tool, but we're limitated by their inverse of a matrix. If we could access all processors units of the world to process one GP, which one would it be? To acess hidden information about nature. Just like when the telescopes around the world united to get a photo of a black hole. 

What I like about this proffessor is that he is so passionate about the topics he teach; he also cares about studings following his words and understanding. For him, there are no stupid questions. I wish I could take a full lecture by him. 

Me recuerda a cuando Nati decia que estaba enamorada de turbo jaja. 

Le debería contar mi idea loca... 


# SINGULARITY AND CONTAINERS

Big problems require big resources. 

When you move the data from CPU to GPU, you have to make sure there's no traffic because the latence time is very high. 

Z.B.; if you are working with batches, you can get the maximal performance by being aware how you can manage the sending of the data and what does the code does in the meantime.

Can you treat regular laptors as HPCs? 

[HPC Centers](https://centers.hpc.mil/users/docs/general/singularity.html) say: Singularity is a tool for running software containers on HPC systems, similar to Docker. Singularity is the first containerization technology supported across DSRC HPC resources. In 2021, development of the container technology known as "Singularity" forked resulting in two similar products.

**Make sure you choose your datatype very carefully.**

* SHOWCASE: how to install any package through singularity. 

# PARALLEL DATA

Back propagation es extremelly expensive; hence, models have been developed to avoid this cost. It's called:
* Extending the Forward Forward Algorithm [arXiv:2307.04205](https://arxiv.org/abs/2307.04205)

**GPUs you use, according to your kernel.**

GPUs use their clock to generate their random number, so if you work with parallelism and your model requires a random seed; then, the input seed would be different for all initializations for the slight time variation. All initialization parameters must be the same, this is why broadcasting is needing when working with data parallelism. 


However, it must be that the sensitivity of the parameters should not impact vastly.  

[Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

[Data Parallelism](https://www.sciencedirect.com/topics/computer-science/data-parallelism)

# GRAPH NEURAL NETWORK

Laplacian graph: graph laplacian spectral analysis yields many information about the structure of a graph, z.B.: the eigenvectors, eigenvalues, partitian in communities.

Graph Neural Networks (GNNs) are a type of deep learning model designed to work with data represented as graphs. Unlike traditional neural networks that operate on vectors or sequences, GNNs can process data with complex relationships and interconnections between entities. This makes them particularly well-suited for analyzing social networks, knowledge graphs, molecular structures, and other graph-structured data. 

Key Concepts:

* Graphs:
    * A graph consists of nodes (representing entities) and edges (representing relationships between entities). 
* Message Passing:
    * GNNs learn by passing messages between neighboring nodes, aggregating information from their connections. 
* Node Embeddings:
    * GNNs learn to represent each node in the graph with a vector (embedding) that captures its features and relationships to other nodes. 
* Graph Embeddings:
    * GNNs can also learn embeddings for entire graphs, which can be used for graph-level tasks like classifying molecules or predicting social network behavior. 
* How they work:
    1. Input:
        * A graph (nodes and edges) and potentially node features (information about each entity). 
    2. Message Passing:
        * Each node aggregates information from its neighbors, updating its own representation. 
    3. Aggregation:
        * The aggregated information is combined with the node's own features to create a new representation for that node. 
    4. Iteration:
        * This process of message passing and aggregation is repeated over multiple layers in the GNN, allowing information to propagate through the graph. 
    5. Output:
    * The final node embeddings can be used for various tasks, such as:
* Node classification: Predicting the category or label of a node. 
* Link prediction: Predicting the existence of a connection between two nodes. 
* Graph classification: Classifying the entire graph based on its structure and node properties. 

Examples of Applications:

* Social Network Analysis: Predicting user behavior, recommending friends, understanding community structures. 
* Knowledge Graphs: Reasoning about facts and relationships in knowledge bases, answering complex questions. 
* Recommender Systems: Suggesting items to users based on their preferences and the preferences of similar users. 
* Drug Discovery: Predicting the properties of molecules, designing new drugs. 
* Materials Science: Predicting the properties of materials, designing new materials with desired characteristics. 
* Recommendation Systems: Suggesting relevant items (e.g., movies, products) to users based on their past interactions and the interactions of similar users. 
* Fraud Detection: Identifying fraudulent transactions in financial networks. 
* Traffic Prediction: Predicting traffic flow and optimizing routes. 

Benefits of GNNs:

* Ability to handle complex relationships:
    * GNNs can capture the intricate connections between entities in a graph, which is not possible with traditional neural networks. 
* Scalability:
    * GNNs can be applied to large graphs with millions or even billions of nodes and edges. 
* Flexibility:
    * GNNs can be adapted to different types of graphs and tasks by modifying the message passing and aggregation functions. 