* Lets start working on task 4.
* I have attached the article (Kuzi et al.) further down

# Task 4: Relevance feedback and Query Expansion (week 20)

The goal of this week is to add a pseudo relevance component to your search engine.

You should use pseudo relevance feedback to expand the query and run it with BM25 and LM. You should tune some of the parameters of pseudo relevance feedback using an evaluation measure of your choice, e.g., if you use RM3 pseudo relevance feedback you can tune only fb terms and fb docs. You should tune with respect to one evaluation measure only (either MRR or NDCG@10).

You should use your single best performing model-index combination as the backbone for the retrieval of the top-K documents to expand the queries and for the final retrieval.

In addition, you should use word embeddings to do query expansion as done by Kuzi et al. A popular word embedding choice is the word2vec pretrained on Google News corpus,19 but you can decide on another embedding.

# Query Expansion Using Word Embeddings

Saar Kuzi
saarkuzi@campus.technion.ac.il

Anna Shtok
annabel@tx.technion.ac.il

Oren Kurland
kurland@ie.technion.ac.il

Technion — Israel Institute of Technology

## ABSTRACT

We present a suite of query expansion methods that are based on word embeddings. Using Word2Vec’s CBOW embedding approach, applied over the entire corpus on which search is performed, we select terms that are semantically related to the query. Our methods either use the terms to expand the original query or integrate them with the effective pseudo-feedback-based relevance model. In the former case, retrieval performance is significantly better than that of using only the query, and in the latter case the performance is significantly better than that of the relevance model.

## 1. INTRODUCTION

Query expansion approaches are intended to ameliorate the vocabulary mismatch between the short query and relevant documents [4]. We present query expansion methods that rely on global, query-independent, analysis of the corpus using word embeddings. Specifically, we use Word2Vec’s Continuous Bag-of-Words (CBOW) approach [11] that represents terms in a vector space based on their co-occurrence in windows of text. Similarities between the term vectors were shown to correspond to semantic similarities [11].

We use the Word2Vec term embeddings to select terms that are similar to the query as a whole or to its constituent terms. The selected terms are used to expand the query in a unigram language model framework. The resultant retrieval performance is significantly better than using the query alone, and a recently proposed translation model based on Word2Vec [21]. In addition, we present an approach to integrating the terms selected using word embeddings with an effective pseudo-feedback-based method: the relevance model [10]. The resultant performance significantly transcends that of the relevance model.

## 2. QUERY EXPANSION MODELS

Our goal is to expand a query q using semantically related terms. To that end, we train offline the Word2Vec Continuous Bag-of-Words (CBOW) model [11] using the entire document corpus D on which search is performed. The model’s goal is to estimate the probability that a term will appear in a position in a text based on terms that appear in a window around this position. As a result, each term in the corpus is represented by a vector embedded in a vector space. Similarities between these vectors were shown to correspond to semantic similarities between the terms [11]. Accordingly, we select terms similar to the query in this vector space.

Our query expansion methods operate in a unigram language model framework. The methods use term scoring functions presented in Section 2.1. Section 2.2 presents a method of using the terms with the highest scores for query expansion. In Section 2.3 we present an approach for integrating the terms selected by our methods with a pseudo-feedback-based relevance model [10].

### 2.1 Term scoring methods

#### 2.1.1 The centroid method

Adding Word2Vec vectors representing terms that constitute an expression often yields a vector that semantically corresponds to the expression [11]. We leverage this observation to score terms for expansion. Herein, ~t denotes the L2-normalized Word2Vec vector representing term t.

We represent q by:

qCent = Σ qi∈q qi

where qi is a query term; this is the query-length scaled centroid of the query terms’ vectors. The score of term t in the collection is:

SCent(t; q) = exp(cos(~t, ~qCent)).        (1)

#### 2.1.2 Fusion-based methods

The Cent approach scores terms by their semantic similarity with the query as a whole. As an alternative, we create for each query term, qi, a list Lqi of its n most similar terms t in the corpus according to cos(~qi, ~t) (cf., [3, 13]); n is a free parameter. These n similarities are softmax-normalized to yield probabilities:

p(t|qi) = exp(cos(~qi, ~t)) / Σt′∈Lqi exp(cos(~qi, ~t′))

for t ∈ Lqi; p(t|qi) = 0 for t ∉ Lqi.

We fuse the resulting term lists using CombSUM, CombMNZ and CombMAX [7]:

SCombSUM(t; q) = Σqi∈q p(t|qi);        (2)

SCombMNZ(t; q) = |{qi ∈ q : p(t|qi) ≠ 0}| Σqi∈q p(t|qi);        (3)

SCombMAX(t; q) = max qi∈q p(t|qi).        (4)

## 2.2 Expanding the query

The ν terms assigned the highest score by a method M ∈ {Cent, CombSUM, CombMNZ, CombMAX} are used for query expansion; ν is a free parameter. We sum-normalize these terms’ scores, SM(t; q), to obtain a probability distribution, p(·|M), that constitutes a unigram language model defined over the vocabulary. Terms not among the ν with the highest scores are assigned zero probability.

We integrate this language model with a language model induced from q. Formally, the maximum likelihood estimate (MLE) of term t with respect to q is:

pMLE(t|q) = tf(t ∈ q) / |q|

where tf(t ∈ q) is the count of t in q and |q| is the query length. The resulting model, Q-M, uses the interpolation parameter λ:

p(t|M, q) = (1 − λ)p(t|M) + λpMLE(t|q).        (5)

## 2.3 Integration with the relevance model

Our premise is that pseudo-feedback-based query expansion approaches are of complementary nature to our global expansion methods that use Word2Vec for corpus analysis. Hence, we now turn to integrate an effective pseudo feedback approach, the relevance model [10], with our methods.

The probability assigned to term t by RM1 is [10]:

p(t|RM1) = Σd∈Dinit p(t|d)p(d|q);        (6)

Dinit is an initial document list described below; p(t|d) is the probability assigned to t by d’s Dirichlet smoothed language model [19]; p(d|q) is d’s normalized query likelihood.

Clipping RM1 by setting to zero the probability of all but the c terms that yield the highest p(t|RM1), and sum-normalizing their probabilities, yields pclip(t|RM1) [1]. Similarly, we use the c terms assigned the highest SM(t; q), where M ∈ {Cent, CombSUM, CombMNZ, CombMAX}, to induce the language model p(·|M) as described above.

The language models are integrated using a parameter α:

p(t|RM, M) = αp(t|M) + (1 − α)pclip(t|RM1).        (7)

We clip this language model to use the ν terms, where ν is a free parameter, assigned the highest p(t|RM, M), yielding pclip(·|RM, M), which is integrated with q:

p(t|RM, M, q) = (1 − λ)pclip(t|RM, M) + λpMLE(t|q).        (8)

These methods are denoted RM-M where M ∈ {Cent, CombSUM, CombMNZ, CombMAX}. If α is set to 0 in Eq. 7, i.e., Word2Vec-based expansion is not used, then Eq. 8 is the relevance model RM3 [1].

## 2.4 Document ranking

Documents are ranked by the cross entropy between a query model and their Dirichlet smoothed language models [19]. We use a few query language models. Using pMLE(·|q) results in an initial standard language-model-based ranking, denoted init, from which the result list Dinit, used for relevance model construction, is derived. We also use the query language models from Eq. 5 and 8, Q-M and RM-M, respectively. Relevance model RM3 [1], which is a special case of Equation 8 as noted above, is used for reference.

## 3. EXPERIMENTAL SETUP

The TREC datasets specified in Table 1 were used for experiments. Titles of TREC topics serve for queries. Krovetz stemming and stopword removal using the INQUERY list were applied to documents and queries. The Indri toolkit (www.lemurproject.org) was used for experiments.

### Table 1: TREC datasets used for experiments

| Collection | TREC disks | # of Docs | Topics |
|---|---|---:|---|
| WSJ | Disks 1&2 | 173,252 | 51–200 |
| AP | Disks 1&3 | 242,918 | 51–200 |
| ROBUST | Disks 4,5-CR | 528,155 | 301–450, 601–700 |
| WT10G | WT10g | 1,692,096 | 451–550 |
| GOV2 | GOV2 | 25,205,179 | 701–850 |

MAP (@1000), precision of the top five documents (p@5), and the reliability of improvement (RI) [14] are used as evaluation measures. RI is:

RI = (|Q+| − |Q−|) / |Q|

Q is the set of queries; Q+ and Q− are the sets queries for which the average precision (AP) is higher and lower, respectively, than that of the initial ranking. RI is a measure of the robustness of a query expansion method. Statistically significant differences of performance are determined using the two-tailed paired t-test at a 95% confidence level.

We train the CBOW model of Word2Vec1 using all documents in each corpus, except for GOV2. For GOV2, due to vocabulary size scale issues, we used the Word2Vec models trained over WT10G2. Documents were processed as described above except that terms longer than 50 characters were excluded due to Word2Vec limitations. Word2Vec incorporates several free parameters: the dimension of the term vectors was set to values in {100, 500}, the number of negative examples is in {5, 10}, and the window size is in {8, 16, 64}; sub-sampling of frequent terms was not performed. All other parameters were set to default values.

The Dirichlet smoothing parameter for document language models is set to 1000 unless otherwise specified. The query weight λ in all expansion models is in {0, 0.2, ..., 1}. The number of terms, ν, used by all query expansion models, is in {10, 25}. The number of terms similar to a single query term, n, see Section 2.1.2, is selected from {50, 100}. For the methods described in Section 2.2, the values of the aforementioned free parameters, including those of Word2Vec, are set using leave-one-out cross validation performed over queries, where MAP serves as the optimization criterion.

The free parameters of the RM-M models, Section 2.3, are set as follows. First, we set the parameters of RM1 per value of c ∈ {50, 100} terms. The other parameters of RM1 are set using leave-one-out cross validation where the maximization criterion is the resultant RM3’s MAP:

1. the Dirichlet smoothing parameter used for document language models for constructing RM1 is in {0, 1000};
2. the number of documents in Dinit from which RM1 is constructed is in {10, 25, 50};
3. the original query weight λ ∈ {0, 0.2, ..., 1}.

The values of all remaining parameters are set using leave-one-out cross validation; here, the MAP of the resultant query expansion method is optimized. The parameters are those of the CBOW model, α ∈ {0, 0.2, ..., 1}, λ ∈ {0, 0.2, ..., 1}, n and c; the latter two are coupled to the same values in the RM-Comb methods.

For RM3, which serves as a baseline, the ranges of the number of expansion terms, the query weight λ, the Dirichlet smoothing parameter used to construct the model, and the number of documents in Dinit from which RM3 is constructed, are as specified above for the RM-M models. Leave-one-out cross validation was applied to set parameter values.

The NTLM translation model serves as an additional baseline [21]. The 10 terms most similar to each query term, based on Word2Vec, were considered as translation terms. Translation probabilities were computed using sum-normalized cosine similarity between a query term and a translation term. The top 1000 documents in the initial ranking were re-ranked using NTLM as we found that ranking all documents in the corpus that contain at least one query term often resulted in inferior performance in four out of five corpora. Following the source code of [21], very frequent or very rare query terms were assigned a 1.0 translation probability3. The free parameters of NTLM, namely those of Word2Vec, were set using leave-one-out cross validation.

1 www.code.google.com/p/word2vec
2 Applying Word2Vec models trained using one corpus on another corpus was shown to be effective for query expansion [17] and translation models [21].
3 The exact criteria for frequency-based term filtering is provided in the source code of [21]; applying the method without this filter or applying the filter on translation terms as well resulted in inferior retrieval effectiveness.

## 4. EXPERIMENTAL RESULTS

### 4.1 Expanding the query

Table 2 presents the performance of the Q-M methods from Section 2.2; these expand the original query with semantically related terms. All these methods outperform, in terms of MAP and p@5, both the initial retrieval and NTLM in the vast majority of cases; the majority of MAP improvements are statistically significant; the Q-M methods are also more robust (RI) than NTLM. In all corpora except for WT10G, the Q-M methods often outperform RM3 in terms of p@5, however rarely statistically significantly. Although RM3 is the best MAP performing expansion method in most cases, the Q-M methods yield effective performance and do not require initial retrieval as RM3.

We also see in Table 2 that no Q-M method dominates the others. The MAP and p@5 of Q-Cent, which performs expansion based on the query as a whole, is almost always statistically indistinguishable from that of the Q-Comb models that use expansions based on individual query terms.

Table 2 shows that NTLM, which uses Word2Vec-based translation probabilities, is almost always outperformed by RM3. Therefore, below we focus on RM3 as a baseline.

### Table 2: Expanding the original query

The best and second best results in each column are boldfaced. The best result in a column is underlined. Statistically significant differences with the initial ranking (init), RM3, NTLM, Q-Cent, Q-CombSUM and Q-CombMNZ are marked with “i”, “r”, “t”, “c”, “s” and “m”, respectively.

| Method | WSJ MAP | WSJ p@5 | WSJ RI | AP MAP | AP p@5 | AP RI | ROBUST MAP | ROBUST p@5 | ROBUST RI | WT10G MAP | WT10G p@5 | WT10G RI | GOV2 MAP | GOV2 p@5 | GOV2 RI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| init | .272 | .489 | - | .216 | .413 | - | .249 | .476 | - | .198 | .355 | - | .299 | .588 | - |
| RM3 | .324i | .520i | .355 | .274i | .435 | .443 | .282i | .483 | .293 | .234i | .390 | .082 | .331i | .596 | .392 |
| NTLM | .277r | .504 | -.073 | .226i,r | .442 | .094 | .241r | .481 | -.241 | .201r | .344 | -.031 | .280i,r | .574 | -.311 |
| Q-Cent | .294i,r,t | .519i | .187 | .243i,r,t | .432 | .289 | .256i,r,t | .488 | .241 | .201r | .348 | .052 | .300r,t | .622i,t | -.041 |
| Q-CombSUM | .295i,r,t | .520i | .227 | .235i,r,c | .447i | .221 | .253r,t | .487 | .137 | .205r | .359 | .134 | .301r,t | .615i,t | -.014 |
| Q-CombMNZ | .291i,r,t | .524i | .273 | .234i,r,c | .451i | .383 | .252i,r,t,c | .476 | .205 | .203r | .353 | .155 | .297r,t,s | .616i,t | -.095 |
| Q-CombMAX | .294i,r,t | .521i | .313 | .237i,r,t | .448i | .282 | .253i,r,t | .491i,m | .237 | .203r | .351 | .124 | .305i,r,t,c,s,m | .631i,r,t | .074 |

### 4.2 Integration with the relevance model

Table 3 presents the performance of the RM-M methods from Section 2.3 which integrate the Word2Vec-based expansion terms with the relevance model. RM-Cent, which integrates terms semantically similar to the query as a whole with the relevance model, performs best in most cases. RM-Cent always outperforms RM3 in terms of MAP and p@5; all MAP improvements are statistically significant. Among the models that utilize expansions based on individual query terms, RM-CombSUM and RM-CombMAX are the most effective: they always outperform RM3 for MAP and p@5. Finally, the robustness (RI) of each of our methods is higher than that of RM3 over at least two out of the five corpora.

### Table 3: Integration with the relevance model

The best result in a column is boldfaced. Statistically significant differences with the initial ranking (init), RM3, RM-Cent, RM-CombSUM and RM-CombMNZ are marked with “i”, “r”, “c”, “s” and “m”, respectively.

| Method | WSJ MAP | WSJ p@5 | WSJ RI | AP MAP | AP p@5 | AP RI | ROBUST MAP | ROBUST p@5 | ROBUST RI | WT10G MAP | WT10G p@5 | WT10G RI | GOV2 MAP | GOV2 p@5 | GOV2 RI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| init | .272 | .489 | - | .216 | .413 | - | .249 | .476 | - | .198 | .355 | - | .299 | .588 | - |
| RM3 | .324i | .520i | .355 | .274i | .435 | .443 | .282i | .483 | .293 | .234i | .390 | .082 | .331i | .596 | .432 |
| RM-Cent | .332i,r | .548i,r | .387 | .282i,r | .464i,r | .416 | .291i,r | .495 | .261 | .240i,r | .396 | .155 | .335i,r | .623r | .392 |
| RM-CombSUM | .325i,c | .533i | .320 | .279i,r | .447 | .483 | .291i,r | .492 | .245 | .236i | .402i | .113 | .335i,r | .618r | .405 |
| RM-CombMNZ | .318i,r,c,s | .523i,c | .247 | .278i,r | .432c | .456 | .289i,r | .483 | .249 | .231i | .396i | .093 | .335i,r | .614r,c | .432 |
| RM-CombMAX | .330i,r,s,m | .541i,r | .373 | .278i | .439c | .483 | .290i,r | .496m | .245 | .237i | .402i | .113 | .336i,r | .622r | .297 |

### 4.3 Further analysis

We use query #546, “recycle cans”, from WT10G to shed some light on the advantages of embedding-based query expansion. The 10 terms assigned the highest probability by RM1, Q-Cent and RM-Cent are:

RM1:
{recycle, can, paint, plastic, paper, accept, steel, contain, material, bottle}

Q-Cent:
{can, recycle, styrofoam, scrap, postconsumer, compost, plastic, trash, cardboard, bottle}

RM-Cent:
{recycle, can, plastic, bottle, paint, styrofoam, scrap, postconsumer, compost, trash}

Boldfaced terms appear in all three lists. Free parameters were fixed to effective values, and no integration with the query was performed so as to focus on the effectiveness of the expansion terms.

The average precision (AP) of RM1, Q-Cent and RM-Cent for the query is 0.084, 0.096 and 0.107, respectively, showing again the merits of integrating Word2Vec-based expansion terms with the relevance model. The three models share 4 out of 10 terms. Among the RM1 terms we find general verbs (“accept”, “contain”) which could cause query drift as is common in pseudo-feedback-based methods. In contrast, all the terms in the Q-Cent list are topical (“scrap”, “styrofoam” and “bottle” are all recycled materials/items). This difference translates to a higher AP of Q-Cent than RM1. Naturally, word-embeddings-based expansion can cause query drift, as is the case for other global expansion methods. Indeed, it was shown that local training of word embeddings using retrieved documents can improve expansion effectiveness [6].

We also found that the top terms of the Word2Vec-based expansion and the relevance model often differ. This further attests to the complementary nature of these expansion approaches leveraged by the RM-M methods.

## 5. RELATED WORK

Word embeddings were used to re-weight the query terms [20] and to compare a query directly with a document [16, 12]. Document embeddings [5, 2] and word-embeddings-based translation models [8, 21] were also used for retrieval. We focused on query expansion and showed that our methods outperform an embedding-based translation model [21]. Embedding-based query expansion was also used in the medical domain [9], but did not outperform retrieval using only the query. Supervised concept-based embedding of n-grams was used for document retrieval [15]. Our methods utilize unsupervised unigram embeddings.

Recently, query expansion approaches conceptually similar to our Q-CombSUM method were proposed [3, 6, 13]: expansion terms are scored using the weighted sum of their embedding-based similarities with query terms. We used additional approaches to integrate these similarities, and combined them with pseudo-feedback-based expansion.

Work on embedding a query language model for query expansion [18] provides formal grounds to using a centroid-based representation of the embedding vectors of the query terms as in our Q-Cent and RM-Cent methods. Our RM-Cent method which integrates the expanded query with a relevance model is different than that which uses an embedding of a relevance model to expand the query [18].

In work to be presented closely in time to ours [17], expansion terms are scored, in spirit, by the arithmetic, similarly to our Q-CombSUM method, and geometric mean of their embedding-based similarities to the query terms; our Q-Cent method was also used; our Q-CombMNZ and Q-CombMAX methods and their RM-M counterparts were not presented. The merits of integrating embedding-based query expansion with a relevance model were demonstrated as in our work; and, the merits of using embeddings for relevance model construction were also shown [17].

## 6. CONCLUSIONS

We demonstrated the merits of query expansion methods that utilize word embeddings induced in a query-independent manner from the entire document corpus. Specifically, our methods utilize the embeddings to expand the query with terms that are semantically related to the query as a whole or to its terms. In addition, we showed the effectiveness of an approach for integrating our query expansion methods with a pseudo-feedback-based query expansion approach; namely, the relevance model.

Acknowledgments. We thank the reviewers for their comments. This work was supported in part by the Israel Science Foundation grant no. 433/12, the Technion-Microsoft Electronic Commerce Research Center, and by the Irwin and Joan Jacobs Fellowship for graduate students.