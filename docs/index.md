***
# Supervised | Unsupervised Learning : Detecting and Evaluating Anomalies in Categorical Data Under Supervised and Unsupervised Settings

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *April 30, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Splitting](#1.4.1)
        * [1.4.2 Data Profiling](#1.4.2)
        * [1.4.3 Category Aggregation and Encoding](#1.4.3)
        * [1.4.4 Outlier and Distributional Shape Analysis](#1.4.4)
        * [1.4.5 Collinearity](#1.4.5)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Premodelling Data Preparation](#1.6)
        * [1.6.1 Preprocessed Data Description](#1.6.1)
        * [1.6.2 Preprocessing Pipeline Development](#1.6.2)
    * [1.7 Model Development With Synthetic Ground Truth Labels](#1.7)
        * [1.7.1 Isolation Forest](#1.7.1)
        * [1.7.2 Local Outlier Factor](#1.7.2)
        * [1.7.3 K-Nearest Neighbors Outlier Score](#1.7.3)
        * [1.7.4 Histogram-Based Outlier Score](#1.7.4)
        * [1.7.5 RuleFit / Tree-Based Methods](#1.7.5)
        * [1.7.6 High Leverage Points via Multiple Correspondence Analysis](#1.7.6)
    * [1.8 Model Development Without Ground Truth Labels](#1.8)
        * [1.8.1 Isolation Forest](#1.8.1)
        * [1.8.2 Local Outlier Factor](#1.8.2)
        * [1.8.3 K-Nearest Neighbors Outlier Score](#1.8.3)
        * [1.8.4 Histogram-Based Outlier Score](#1.8.4)
        * [1.8.5 RuleFit / Tree-Based Methods](#1.8.5)
        * [1.8.6 High Leverage Points via Multiple Correspondence Analysis](#1.8.6)
    * [1.9 Consolidated Findings](#1.9)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores various **Outlier Detection** techniques specifically tailored for datasets with purely categorical features, utilizing multiple tools and libraries available in <mark style="background-color: #CCECFF"><b>Python</b></mark>. The analysis was carried out in two distinct experimental settings: a **Supervised Setting** where synthetic outlier ground truth labels are available, and an **Unsupervised Setting** where no labels are assumed. The methods applied in both settings span a diverse range of non-deep learning techniques, including classical statistical, distance-based, and pattern-based models including: **Histogram-Based Outlier Score (HBOS)**, **K-Nearest Neighbors (KNN)**, **Clustering-Based Local Outlier Factor (CBLOF)**, **Isolation Forest**, **Multiple Correspondence Analysis + Mahalanobis Distance**, **K-Modes with Cluster Distance** and **Association Rule Violation Count**. In the supervised setting, outlier detection methods were evaluated using traditional classification metrics such as the **Area Under the Receiver Operating Characteristic Curve(AUROC)**, **F1-Score**, and **Precision@N** to assess their ability to distinguish true outliers from normal observations. In contrast, the unsupervised setting employed label-agnostic evaluation strategies, leveraging internal and score-based metrics such as **Outlier Score Entropy**, **Score Variance**, **Silhouette Score on Outlier Scores**, and **Clustering Stability**. These were complemented by **t-SNE Visualizations** to assess the score separability and clustering quality across methods. This dual-framework approach allows for a comprehensive understanding of how each method performs under both label-available and label-free conditions, providing a rigorous basis for outlier detection in categorical data contexts. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[Outlier Detection](https://link.springer.com/book/10.1007/978-3-319-47578-3) in datasets composed exclusively of categorical variables poses unique challenges because most traditional techniques rely on notions like distance, density, or variance—concepts naturally defined in numerical spaces. In contrast, categorical data lacks inherent numeric relationships or ordering, making it non-trivial to apply distance-based or distribution-based methods directly. However, there are still structured and principled approaches to identify anomalies in such datasets. At its core, an outlier in categorical data is an observation that exhibits a rare or unexpected combination of attribute levels. While each individual variable may contain common categories, their joint configuration may be highly improbable, making the observation an outlier even if no single variable is unusual in isolation. The most basic yet powerful approach begins by examining frequency distributions by way of tallying the count of each unique record configuration and flagging those that occur rarely or not at all in the rest of the data as potential anomalies. To go a step further, one can estimate probabilities of individual variable levels and their joint probabilities across combinations of features. This idea underlies models where the joint likelihood of feature values is used as an outlier score. However, due to the curse of dimensionality — even with a moderate number of variables — the number of possible category combinations grows exponentially, making pure joint frequency estimation unreliable in high dimensions. This motivates the use of dimensionality reduction techniques which maps categorical data into a continuous latent space. In this lower-dimensional space, standard outlier detection techniques can be applied more reliably. In other cases, categorical variables are transformed using encoding schemes to enable the use of well-established numeric algorithms. For instance, one-hot encoding represents each level of a categorical variable as a binary vector, while ordinal encoding assigns arbitrary numeric values to categories. More advanced encodings like entity embeddings can preserve semantic similarity between categories learned via neural networks. Once encoded, various outlier detection methods can be applied, provided the encoding faithfully retains category relationships. An alternative family of methods relies on clustering to group similar categorical records. Outliers are identified either as records not belonging to any dense cluster, or as records that lie far from their cluster's central profile. Finally, rule-based outlier detection extract conditional patterns and flag records that violate these frequent rules. In conclusion, while outlier detection in categorical data is fundamentally different from numeric data due to the lack of inherent distances or ordering, a combination of probability modeling, encoding and transformation, rule learning, and clustering can be used to detect anomalies effectively. The choice of method depends on the structure of the data, the cardinality of features, and whether interpretability or predictive performance is prioritized.

[Outlier Detection In Categorical Data With Ground Truth Labels](https://link.springer.com/book/10.1007/978-3-319-47578-3) becomes a supervised classification task. In this setting, the goal is not just to detect anomalies, but to train models that can predict outlier status reliably, based on known labeled examples. This scenario is relatively rare in real-world applications, but it allows for robust evaluation and algorithm benchmarking. To begin, each observation in the dataset is tagged as either an "inlier" or an "outlier". This label can be derived from domain expertise, human annotation, or synthetic injection of anomalies for experimental setups. The categorical features are then encoded using techniques such as one-hot, ordinal, or entity embeddings so that they can be processed by standard supervised learning models or outlier scoring algorithms. Outlier detection methods are then trained on these encoded features. Even though these methods are typically unsupervised, in this scenario, their outputs—namely, the anomaly scores — can be evaluated using the known labels. Evaluation metrics for this setting include: Precision@N (a variation of precision that is particularly useful in ranking problems like outlier detection for evaluating the top N most confident predictions rather than all predictions), F1-Score (harmonic mean of precision and recall for balancing both metrics, particularly useful when there's a tradeoff), AUROC (Area Under the Receiver Operating Characteristic Curve) (evaluates the model’s ability to distinguish between inliers and outliers across various thresholds, where a higher value indicates better separability). This setup allows for quantitative comparison of different models and hyperparameters. Because true outliers are known, models can be ranked, tuned, and selected with confidence.

[Outlier Detection In Categorical Data Without Ground Truth Labels](https://link.springer.com/book/10.1007/978-3-319-47578-3) is a purely unsupervised learning task. This introduces significant challenges: since the true identity of outliers is unknown, models cannot be trained or validated using standard supervised metrics. Instead, evaluation shifts toward the intrinsic structure of the data and the behavior of anomaly scores. To begin, categorical data must be carefully encoded using strategies that retain their semantic meaning. Common encoding methods include one-hot encoding (to preserve disjoint category identity), ordinal encoding (for ordered categories), and entity embeddings (to capture latent similarity among categories). Once the data is numerically represented, various unsupervised algorithms can be applied to compute outlier scores. These scores reflect the degree of "outlierness" of each observation based on algorithm-specific logic such as isolation depth, density deviation, or cluster distance. In the absence of ground truth labels, the quality of these scores is evaluated using unsupervised score-based metrics. These metrics help assess whether the algorithm has meaningfully differentiated outliers from inliers in a data-driven way including Outlier Score Entropy (measures the unpredictability or uniformity in the distribution of outlier scores where a very low entropy may indicate that the model is not distinguishing between normal and anomalous observations), score variance (examines the spread of anomaly scores across all observations where a low variance suggests that the model assigns similar scores to most points, possibly indicating insensitivity to actual structure), silhouette score on outlier scores (clusters the outlier scores themselves into two or more groups and evaluates how well-separated these clusters are with a high silhouette score suggesting that the model produces score groupings that align with distinguishable data behavior, reinforcing the meaningfulness of its outlier assignments) and clustering stability (assesses how consistent the outlier groupings remain when the input data or model parameters are slightly perturbed with low variability across runs implying that the model is robust and not overly sensitive to sampling artifacts, which boosts confidence in the flagged anomalies). These unsupervised evaluation techniques offer a practical lens through which model performance can be judged, even in the complete absence of labeled anomalies. When multiple models consistently flag the same observations as anomalous, or when scores exhibit structured and stable separation, this further validates the relevance of the identified outliers. Ultimately, there is no single "correct" model in unsupervised outlier detection without ground truth. Instead, a combination of score distribution analysis, clustering behavior, consistency checks, and domain interpretability is used to triangulate the credibility of the results. This makes practitioner insight and a deep understanding of the domain especially important when applying these methods to categorical data.



## 1.1. Data Background <a class="anchor" id="1.1"></a>

## 1.2. Data Description <a class="anchor" id="1.2"></a>

## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Data Splitting <a class="anchor" id="1.4.1"></a>

### 1.4.2 Data Profiling <a class="anchor" id="1.4.2"></a>

### 1.4.3 Category Aggregration and Encoding <a class="anchor" id="1.4.3"></a>

### 1.4.4 Outlier and Distributional Shape Analysis <a class="anchor" id="1.4.4"></a>

### 1.4.5 Collinearity <a class="anchor" id="1.4.5"></a>

## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

## 1.6. Premodelling Data Preparation <a class="anchor" id="1.6"></a>

### 1.6.1 Preprocessed Data Description <a class="anchor" id="1.6.1"></a>

### 1.6.2 Preprocessing Pipeline Development <a class="anchor" id="1.6.2"></a>

## 1.7. Model Development With Synthetic Ground Truth Labels <a class="anchor" id="1.7"></a>

### 1.7.1 Isolation Forest <a class="anchor" id="1.7.1"></a>

### 1.7.2 Local Outlier Factor <a class="anchor" id="1.7.2"></a>

### 1.7.3 K-Nearest Neighbors Outlier Score <a class="anchor" id="1.7.3"></a>

### 1.7.4 Histogram-Based Outlier Score <a class="anchor" id="1.7.4"></a>

### 1.7.5 RuleFit / Tree-Based Methods <a class="anchor" id="1.7.5"></a>

### 1.7.6 High Leverage Points via Multiple Correspondence Analysis <a class="anchor" id="1.7.6"></a>

## 1.8. Model Development Without Ground Truth Labels <a class="anchor" id="1.8"></a>

### 1.8.1 Isolation Forest <a class="anchor" id="1.8.1"></a>

### 1.8.2 Local Outlier Factor <a class="anchor" id="1.8.2"></a>

### 1.8.3 K-Nearest Neighbors Outlier Score <a class="anchor" id="1.8.3"></a>

### 1.8.4 Histogram-Based Outlier Score <a class="anchor" id="1.8.4"></a>

### 1.8.5 RuleFit / Tree-Based Methods <a class="anchor" id="1.8.5"></a>

### 1.8.6 High Leverage Points via Multiple Correspondence Analysis <a class="anchor" id="1.8.6"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Applied Predictive Modeling](http://appliedpredictivemodeling.com/) by Max Kuhn and Kjell Johnson
* **[Book]** [An Introduction to Statistical Learning](https://www.statlearning.com/) by Gareth James, Daniela Witten, Trevor Hastie and Rob Tibshirani
* **[Book]** [Outlier Analysis](https://link.springer.com/book/10.1007/978-3-319-47578-3) by Charu Aggarwal 
* **[Book]** [Beginning Anomaly Detection Using Python-Based Deep Learning: With Keras and PyTorch](https://link.springer.com/book/10.1007/978-1-4842-5177-5) by Sridhar Alla and Suman Kalyan Adari
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.image](https://matplotlib.org/stable/api/image_api.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.offsetbox](https://matplotlib.org/stable/api/offsetbox_api.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) by Schttps://scikit-learn.org/stable/api/sklearn.neighbors.htmlikit-Learn Team
* **[Python Library API]** [sklearn.neighbors](https://scikit-learn.org/stable/api/sklearn.neighbors.html) by Scikit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [pyod](https://pyod.readthedocs.io/en/latest/index.html) by PyOD Team
* **[Python Library API]** [prince](https://github.com/MaxHalford/prince) by Prince Team
* **[Article]** [Awesome Outlier Detection Resources](https://github.com/durgeshsamariya/awesome-outlier-detection-resources) by Durgesh Samariya (GitHub.Com)
* **[Video Tutorial]** [Complete Anomaly Detection Tutorials Machine Learning And Its Types With Implementation](https://www.youtube.com/watch?v=OS9xRGKfx4E&t=1790s) by Krish Naik (YouTube)
* **[Course]** [DataCamp Python Data Analyst Certificate](https://app.datacamp.com/learn/career-tracks/data-analyst-with-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Associate Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/associate-data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Python Data Scientist Certificate](https://app.datacamp.com/learn/career-tracks/data-scientist-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Engineer Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Machine Learning Scientist Certificate](https://app.datacamp.com/learn/career-tracks/machine-learning-scientist-with-python) by DataCamp Team (DataCamp)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)




```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

