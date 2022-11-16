# NLP_debias_project

Our project replicates and extends the claims made in the paper Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. This paper highlights the biases that exist in the data and how the machine learning models run the risk of amplifying them. Wordembedding is a popular framework to represent text data as vectors and most of the machine/deep learning models these days make extensive use of it. Word embedding represents text as a d-dimensional vector. So, all those words which have similar semantic meaning ends up close to each other in this d-dimensional vector space

# Authors
Madhuri Pujari | Harish Chauhan | Dhruv Agarwal


# Dataset
w2v_gnews_small.txt -  w2vNEWS (Word2Vec embedding trained on a corpus of Google news texts) 
This is a word embeddings trained on Google News articles which exhibit female/male gender stereotypes to a disturbing extent. This raises concerns because their widespread use, as we describe, often tends to amplify these biases. 
Geometrically, gender bias is first shown to be captured by a direction in the word embedding. 
Second, gender neutral words are shown to be linearly separable from gender definition words in the word embedding. 

# Debias Algorithm
The paper utilizes following methods to success fully dampen the effect of bias in the embedding while still preserving its useful properties such asthe ability to cluster related concepts and to solve analogy tasks.
  1. **Identify gender bias subspace**, the authors use Principal Component Analysis(PCA) on 10 gender pair difference vectors and show that the majority of variance in these vectors is only along one principal axis. The first eigenvalue is significantly larger than the rest.
  2. **Hard de-biasing (neutralize and equalize)** To lessen the impact of biases, the authors introduce a method viz. neutralize and equalize. It removes gender neutral words from gender subspace and make them equidistant outside this subspace.
  3. **Soft bias correction** Equalize removes certain distinctions that are valuable in certain applications. The Soften algorithm reduces the differences between these sets while maintaining as much similarity to the original emedding as possible, with a parameter that controls this trade-off.


