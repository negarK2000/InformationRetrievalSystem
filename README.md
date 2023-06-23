# Information Retrieval System (From Scratch)
Fall 2021

## A Search Engine for Retrieving Articles from a Persian News Website

**This project was developed to create an Information Retrieval system specifically for Persian news. However, with minimal adjustments, it can be adapted to any language or database.**

The system provides ranked retrieval of documents based on their relevance to user queries.

---

### Phase 1: Simple Information Retrieval Model

In this phase of the project, we establish a simple yet effective information retrieval model. The following tasks are involved:

- **1.1 - Document Preprocessing**
  - Token extraction
  - Text normalization
  - Removal of stop words (using this [dataset](https://github.com/kharazi/persian-stopwords) for stop words, with additional words added)
  - Stemming

- **1.2 - Creation of Position and Inverted Indexes**
  
- **1.3 - Answering User Queries**
  
- **1.4 - Integration with the Database**

---

### Phase 2 - Version 1: Enhanced Information Retrieval Model with Advanced Functionalities

In this stage, we expand the information retrieval model by representing documents in a vector format. This allows us to rank search results based on their relevance to user queries. The following steps are undertaken:

- **2.1.1 - Modeling Documents in Vector Space:** TF-IDF Weighting Method

  - The equation used: 𝑓𝑖𝑑𝑓(𝑡,𝑑,𝐷) = 𝑡𝑓(𝑡,𝑑) × 𝑖𝑑𝑓(𝑡,𝐷) = (1 + log(𝑓_𝑡,𝑑)) × log(𝑁/𝑛_𝑡)

- **2.1.2 - Answering Queries in Vector Space:** Cosine Similarity Measure
  
  - The equation used: 𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦(𝑎,𝑏) = cos(𝜃) = 𝑎.𝑏 / (‖𝑎‖ × ‖𝑏‖) = Σ(𝑎_𝑖 × 𝑏_𝑖) / (sqr(Σ(𝑎_𝑖^2)) × sqr(Σ(𝑏_𝑖^2)))

- **2.1.3 - Enhancing Query Processing Speed:** Champion List Technique

### Phase 2 - Version 2: Document Representation Using Word Embedding

- **2.2.1 - Document Representation:** Word2Vec Skip-Gram Model

  - We obtained the document representation using the Word2Vec skip-gram model (Utilizing the gensim library).

- **2.2.2 - Query Representation**

- **2.2.3 - Performance Analysis of the Information Retrieval Model and Reporting**

  - Mean Reciprocal Rank
  - Mean Average Precision
  - Precision@k (k = 1 to 5)

---

### Phase 3: Implementation of Clustering, Categorization, and Retrieval Based on Cluster/Category

In this phase, we tackle the scalability challenge of the search engine by incorporating clustering techniques. Instead of comparing queries with all documents, we compare the feature vector of the query with documents within specific clusters. Furthermore, we implement news classification, mapping articles into categories such as sports, economic, political, health, and cultural. This enables efficient determination of news categories for search results.

- **3.1 - Clustering:** K-means Algorithm and Optimal Cluster Selection based on RSS Criteria

- **3.2 - Grouping:** k Nearest Neighbor Algorithm with Cross-Validation Method to Determine Optimal k

---

#### Ranked Retrieval
The ranked retrieval process employs TF-IDF vectors to represent documents and queries, calculating their similarity using cosine similarity. News articles are retrieved from inverted indices constructed from the dataset collections.
