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
        * [1.4.1 Ordinal Binning](#1.4.1)
        * [1.4.2 Category Aggregration and Encoding](#1.4.2)
        * [1.4.3 Synthetic Outlier Labeling via Frequency-Based Tagging](#1.4.3)
        * [1.4.4 Data Splitting](#1.4.4)
        * [1.4.5 Data Preparation](#1.4.5)
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

An open [Thyroid Disease Dataset](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Jai Naru](https://www.kaggle.com/jainaru) and [Abuchi Onwuegbusi](https://www.kaggle.com/datasets/abuchionwuegbusi/thyroid-cancer-recurrence-prediction/data)) was used for the analysis as consolidated from the following primary sources: 
1. Reference Repository entitled **Differentiated Thyroid Cancer Recurrence** from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)
2. Research Paper entitled **Machine Learning for Risk Stratification of Thyroid Cancer Patients: a 15-year Cohort Study** from the [European Archives of Oto-Rhino-Laryngology](https://link.springer.com/article/10.1007/s00405-023-08299-w)

This study hypothesized that the various clinicopathological characteristics influence differentiated thyroid cancer recurrence between patients.

The dichotomous categorical variable for the study is:
* <span style="color: #FF0000">Recurred</span> - Status of the patient (Yes, Recurrence of differentiated thyroid cancer | No, No recurrence of differentiated thyroid cancer)

The predictor variables for the study are:
* <span style="color: #FF0000">Age</span> - Patient's age (Years)
* <span style="color: #FF0000">Gender</span> - Patient's sex (M | F)
* <span style="color: #FF0000">Smoking</span> - Indication of smoking (Yes | No)
* <span style="color: #FF0000">Hx Smoking</span> - Indication of smoking history (Yes | No)
* <span style="color: #FF0000">Hx Radiotherapy</span> - Indication of radiotherapy history for any condition (Yes | No)
* <span style="color: #FF0000">Thyroid Function</span> - Status of thyroid function (Clinical Hyperthyroidism, Hypothyroidism | Subclinical Hyperthyroidism, Hypothyroidism | Euthyroid)
* <span style="color: #FF0000">Physical Examination</span> - Findings from physical examination including palpation of the thyroid gland and surrounding structures (Normal | Diffuse Goiter | Multinodular Goiter | Single Nodular Goiter Left, Right)
* <span style="color: #FF0000">Adenopathy</span> - Indication of enlarged lymph nodes in the neck region (No | Right | Extensive | Left | Bilateral | Posterior)
* <span style="color: #FF0000">Pathology</span> - Specific thyroid cancer type as determined by pathology examination of biopsy samples (Follicular | Hurthel Cell | Micropapillary | Papillary)
* <span style="color: #FF0000">Focality</span> - Indication if the cancer is limited to one location or present in multiple locations (Uni-Focal | Multi-Focal)
* <span style="color: #FF0000">Risk</span> - Risk category of the cancer based on various factors, such as tumor size, extent of spread, and histological type (Low | Intermediate | High)
* <span style="color: #FF0000">T</span> - Tumor classification based on its size and extent of invasion into nearby structures (T1a | T1b | T2 | T3a | T3b | T4a | T4b)
* <span style="color: #FF0000">N</span> - Nodal classification indicating the involvement of lymph nodes (N0 | N1a | N1b)
* <span style="color: #FF0000">M</span> - Metastasis classification indicating the presence or absence of distant metastases (M0 | M1)
* <span style="color: #FF0000">Stage</span> - Overall stage of the cancer, typically determined by combining T, N, and M classifications (I | II | III | IVa | IVb)
* <span style="color: #FF0000">Response</span> - Cancer's response to treatment (Biochemical Incomplete | Indeterminate | Excellent | Structural Incomplete)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The initial tabular dataset was comprised of 383 observations and 17 variables (including 1 target and 16 predictors).
    * **383 rows** (observations)
    * **17 columns** (variables)
        * **1/17 target** (categorical)
             * <span style="color: #FF0000">Recurred</span>
        * **1/17 predictor** (numeric)
             * <span style="color: #FF0000">Age</span>
        * **16/17 predictor** (categorical)
             * <span style="color: #FF0000">Gender</span>
             * <span style="color: #FF0000">Smoking</span>
             * <span style="color: #FF0000">Hx_Smoking</span>
             * <span style="color: #FF0000">Hx_Radiotherapy</span>
             * <span style="color: #FF0000">Thyroid_Function</span>
             * <span style="color: #FF0000">Physical_Examination</span>
             * <span style="color: #FF0000">Adenopathy</span>
             * <span style="color: #FF0000">Pathology</span>
             * <span style="color: #FF0000">Focality</span>
             * <span style="color: #FF0000">Risk</span>
             * <span style="color: #FF0000">T</span>
             * <span style="color: #FF0000">N</span>
             * <span style="color: #FF0000">M</span>
             * <span style="color: #FF0000">Stage</span>
             * <span style="color: #FF0000">Response</span>



```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import itertools
import os
import pickle
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OrdinalEncoder
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency
from scipy.spatial.distance import cdist

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import TruncatedSVD

from pyod.models.knn import KNN
from pyod.models.hbos import HBOS

import prince

```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
DATASETS_FINAL_PATH = r"datasets\final\complete"
DATASETS_FINAL_TRAIN_PATH = r"datasets\final\train"
DATASETS_FINAL_TRAIN_FEATURES_PATH = r"datasets\final\train\features"
DATASETS_FINAL_TRAIN_TARGET_PATH = r"datasets\final\train\target"
DATASETS_FINAL_VALIDATION_PATH = r"datasets\final\validation"
DATASETS_FINAL_VALIDATION_FEATURES_PATH = r"datasets\final\validation\features"
DATASETS_FINAL_VALIDATION_TARGET_PATH = r"datasets\final\validation\target"
DATASETS_FINAL_TEST_PATH = r"datasets\final\test"
DATASETS_FINAL_TEST_FEATURES_PATH = r"datasets\final\test\features"
DATASETS_FINAL_TEST_TARGET_PATH = r"datasets\final\test\target"
MODELS_PATH = r"models"

```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
thyroid_cancer = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "Thyroid_Diff.csv"))

```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(thyroid_cancer.shape)

```

    Dataset Dimensions: 
    


    (383, 17)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(thyroid_cancer.dtypes)

```

    Column Names and Data Types:
    


    Age                      int64
    Gender                  object
    Smoking                 object
    Hx Smoking              object
    Hx Radiotherapy         object
    Thyroid Function        object
    Physical Examination    object
    Adenopathy              object
    Pathology               object
    Focality                object
    Risk                    object
    T                       object
    N                       object
    M                       object
    Stage                   object
    Response                object
    Recurred                object
    dtype: object



```python
##################################
# Renaming and standardizing the column names
# to replace blanks with undercores
##################################
thyroid_cancer.columns = thyroid_cancer.columns.str.replace(" ", "_")

```


```python
##################################
# Taking a snapshot of the dataset
##################################
thyroid_cancer.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Selecting categorical columns (both object and categorical types)
# and listing the unique categorical levels
##################################
cat_cols = thyroid_cancer.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    print(f"Categorical | Object Column: {col}")
    print(thyroid_cancer[col].unique())  
    print("-" * 40)
    
```

    Categorical | Object Column: Gender
    ['F' 'M']
    ----------------------------------------
    Categorical | Object Column: Smoking
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Hx_Smoking
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Hx_Radiotherapy
    ['No' 'Yes']
    ----------------------------------------
    Categorical | Object Column: Thyroid_Function
    ['Euthyroid' 'Clinical Hyperthyroidism' 'Clinical Hypothyroidism'
     'Subclinical Hyperthyroidism' 'Subclinical Hypothyroidism']
    ----------------------------------------
    Categorical | Object Column: Physical_Examination
    ['Single nodular goiter-left' 'Multinodular goiter'
     'Single nodular goiter-right' 'Normal' 'Diffuse goiter']
    ----------------------------------------
    Categorical | Object Column: Adenopathy
    ['No' 'Right' 'Extensive' 'Left' 'Bilateral' 'Posterior']
    ----------------------------------------
    Categorical | Object Column: Pathology
    ['Micropapillary' 'Papillary' 'Follicular' 'Hurthel cell']
    ----------------------------------------
    Categorical | Object Column: Focality
    ['Uni-Focal' 'Multi-Focal']
    ----------------------------------------
    Categorical | Object Column: Risk
    ['Low' 'Intermediate' 'High']
    ----------------------------------------
    Categorical | Object Column: T
    ['T1a' 'T1b' 'T2' 'T3a' 'T3b' 'T4a' 'T4b']
    ----------------------------------------
    Categorical | Object Column: N
    ['N0' 'N1b' 'N1a']
    ----------------------------------------
    Categorical | Object Column: M
    ['M0' 'M1']
    ----------------------------------------
    Categorical | Object Column: Stage
    ['I' 'II' 'IVB' 'III' 'IVA']
    ----------------------------------------
    Categorical | Object Column: Response
    ['Indeterminate' 'Excellent' 'Structural Incomplete'
     'Biochemical Incomplete']
    ----------------------------------------
    Categorical | Object Column: Recurred
    ['No' 'Yes']
    ----------------------------------------
    


```python
##################################
# Correcting a category level
##################################
thyroid_cancer["Pathology"] = thyroid_cancer["Pathology"].replace("Hurthel cell", "Hurthle Cell")

```


```python
##################################
# Setting the levels of the categorical variables
##################################
thyroid_cancer['Recurred'] = thyroid_cancer['Recurred'].astype('category')
thyroid_cancer['Recurred'] = thyroid_cancer['Recurred'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Gender'] = thyroid_cancer['Gender'].astype('category')
thyroid_cancer['Gender'] = thyroid_cancer['Gender'].cat.set_categories(['M', 'F'], ordered=True)
thyroid_cancer['Smoking'] = thyroid_cancer['Smoking'].astype('category')
thyroid_cancer['Smoking'] = thyroid_cancer['Smoking'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Hx_Smoking'] = thyroid_cancer['Hx_Smoking'].astype('category')
thyroid_cancer['Hx_Smoking'] = thyroid_cancer['Hx_Smoking'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Hx_Radiotherapy'] = thyroid_cancer['Hx_Radiotherapy'].astype('category')
thyroid_cancer['Hx_Radiotherapy'] = thyroid_cancer['Hx_Radiotherapy'].cat.set_categories(['No', 'Yes'], ordered=True)
thyroid_cancer['Thyroid_Function'] = thyroid_cancer['Thyroid_Function'].astype('category')
thyroid_cancer['Thyroid_Function'] = thyroid_cancer['Thyroid_Function'].cat.set_categories(['Euthyroid', 'Subclinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Clinical Hyperthyroidism'], ordered=True)
thyroid_cancer['Physical_Examination'] = thyroid_cancer['Physical_Examination'].astype('category')
thyroid_cancer['Physical_Examination'] = thyroid_cancer['Physical_Examination'].cat.set_categories(['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right', 'Multinodular goiter', 'Diffuse goiter'], ordered=True)
thyroid_cancer['Adenopathy'] = thyroid_cancer['Adenopathy'].astype('category')
thyroid_cancer['Adenopathy'] = thyroid_cancer['Adenopathy'].cat.set_categories(['No', 'Left', 'Right', 'Bilateral', 'Posterior', 'Extensive'], ordered=True)
thyroid_cancer['Pathology'] = thyroid_cancer['Pathology'].astype('category')
thyroid_cancer['Pathology'] = thyroid_cancer['Pathology'].cat.set_categories(['Hurthle Cell', 'Follicular', 'Micropapillary', 'Papillary'], ordered=True)
thyroid_cancer['Focality'] = thyroid_cancer['Focality'].astype('category')
thyroid_cancer['Focality'] = thyroid_cancer['Focality'].cat.set_categories(['Uni-Focal', 'Multi-Focal'], ordered=True)
thyroid_cancer['Risk'] = thyroid_cancer['Risk'].astype('category')
thyroid_cancer['Risk'] = thyroid_cancer['Risk'].cat.set_categories(['Low', 'Intermediate', 'High'], ordered=True)
thyroid_cancer['T'] = thyroid_cancer['T'].astype('category')
thyroid_cancer['T'] = thyroid_cancer['T'].cat.set_categories(['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'], ordered=True)
thyroid_cancer['N'] = thyroid_cancer['N'].astype('category')
thyroid_cancer['N'] = thyroid_cancer['N'].cat.set_categories(['N0', 'N1a', 'N1b'], ordered=True)
thyroid_cancer['M'] = thyroid_cancer['M'].astype('category')
thyroid_cancer['M'] = thyroid_cancer['M'].cat.set_categories(['M0', 'M1'], ordered=True)
thyroid_cancer['Stage'] = thyroid_cancer['Stage'].astype('category')
thyroid_cancer['Stage'] = thyroid_cancer['Stage'].cat.set_categories(['I', 'II', 'III', 'IVA', 'IVB'], ordered=True)
thyroid_cancer['Response'] = thyroid_cancer['Response'].astype('category')
thyroid_cancer['Response'] = thyroid_cancer['Response'].cat.set_categories(['Excellent', 'Structural Incomplete', 'Biochemical Incomplete', 'Indeterminate'], ordered=True)

```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(thyroid_cancer.describe(include='number').transpose())

```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>383.0</td>
      <td>40.866841</td>
      <td>15.134494</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>37.0</td>
      <td>51.0</td>
      <td>82.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variables
##################################
print('Categorical Variable Summary:')
display(thyroid_cancer.describe(include='category').transpose())

```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gender</th>
      <td>383</td>
      <td>2</td>
      <td>F</td>
      <td>312</td>
    </tr>
    <tr>
      <th>Smoking</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>334</td>
    </tr>
    <tr>
      <th>Hx_Smoking</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>355</td>
    </tr>
    <tr>
      <th>Hx_Radiotherapy</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>376</td>
    </tr>
    <tr>
      <th>Thyroid_Function</th>
      <td>383</td>
      <td>5</td>
      <td>Euthyroid</td>
      <td>332</td>
    </tr>
    <tr>
      <th>Physical_Examination</th>
      <td>383</td>
      <td>5</td>
      <td>Single nodular goiter-right</td>
      <td>140</td>
    </tr>
    <tr>
      <th>Adenopathy</th>
      <td>383</td>
      <td>6</td>
      <td>No</td>
      <td>277</td>
    </tr>
    <tr>
      <th>Pathology</th>
      <td>383</td>
      <td>4</td>
      <td>Papillary</td>
      <td>287</td>
    </tr>
    <tr>
      <th>Focality</th>
      <td>383</td>
      <td>2</td>
      <td>Uni-Focal</td>
      <td>247</td>
    </tr>
    <tr>
      <th>Risk</th>
      <td>383</td>
      <td>3</td>
      <td>Low</td>
      <td>249</td>
    </tr>
    <tr>
      <th>T</th>
      <td>383</td>
      <td>7</td>
      <td>T2</td>
      <td>151</td>
    </tr>
    <tr>
      <th>N</th>
      <td>383</td>
      <td>3</td>
      <td>N0</td>
      <td>268</td>
    </tr>
    <tr>
      <th>M</th>
      <td>383</td>
      <td>2</td>
      <td>M0</td>
      <td>365</td>
    </tr>
    <tr>
      <th>Stage</th>
      <td>383</td>
      <td>5</td>
      <td>I</td>
      <td>333</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>383</td>
      <td>4</td>
      <td>Excellent</td>
      <td>208</td>
    </tr>
    <tr>
      <th>Recurred</th>
      <td>383</td>
      <td>2</td>
      <td>No</td>
      <td>275</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variable levels
# based on the ordered categories
##################################
ordered_cat_cols = thyroid_cancer.select_dtypes(include=["category"]).columns
for col in ordered_cat_cols:
    print(f"Column: {col}")
    print("Absolute Frequencies:")
    print(thyroid_cancer[col].value_counts().reindex(thyroid_cancer[col].cat.categories))
    print("\nNormalized Frequencies:")
    print(thyroid_cancer[col].value_counts(normalize=True).reindex(thyroid_cancer[col].cat.categories))
    print("-" * 50)
    
```

    Column: Gender
    Absolute Frequencies:
    M     71
    F    312
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M    0.185379
    F    0.814621
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Smoking
    Absolute Frequencies:
    No     334
    Yes     49
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.872063
    Yes    0.127937
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Smoking
    Absolute Frequencies:
    No     355
    Yes     28
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.926893
    Yes    0.073107
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Radiotherapy
    Absolute Frequencies:
    No     376
    Yes      7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.981723
    Yes    0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Thyroid_Function
    Absolute Frequencies:
    Euthyroid                      332
    Subclinical Hypothyroidism      14
    Subclinical Hyperthyroidism      5
    Clinical Hypothyroidism         12
    Clinical Hyperthyroidism        20
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Euthyroid                      0.866841
    Subclinical Hypothyroidism     0.036554
    Subclinical Hyperthyroidism    0.013055
    Clinical Hypothyroidism        0.031332
    Clinical Hyperthyroidism       0.052219
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Physical_Examination
    Absolute Frequencies:
    Normal                           7
    Single nodular goiter-left      89
    Single nodular goiter-right    140
    Multinodular goiter            140
    Diffuse goiter                   7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Normal                         0.018277
    Single nodular goiter-left     0.232376
    Single nodular goiter-right    0.365535
    Multinodular goiter            0.365535
    Diffuse goiter                 0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Adenopathy
    Absolute Frequencies:
    No           277
    Left          17
    Right         48
    Bilateral     32
    Posterior      2
    Extensive      7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No           0.723238
    Left         0.044386
    Right        0.125326
    Bilateral    0.083551
    Posterior    0.005222
    Extensive    0.018277
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Pathology
    Absolute Frequencies:
    Hurthle Cell       20
    Follicular         28
    Micropapillary     48
    Papillary         287
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Hurthle Cell      0.052219
    Follicular        0.073107
    Micropapillary    0.125326
    Papillary         0.749347
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Focality
    Absolute Frequencies:
    Uni-Focal      247
    Multi-Focal    136
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Uni-Focal      0.644909
    Multi-Focal    0.355091
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Risk
    Absolute Frequencies:
    Low             249
    Intermediate    102
    High             32
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Low             0.650131
    Intermediate    0.266319
    High            0.083551
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: T
    Absolute Frequencies:
    T1a     49
    T1b     43
    T2     151
    T3a     96
    T3b     16
    T4a     20
    T4b      8
    Name: count, dtype: int64
    
    Normalized Frequencies:
    T1a    0.127937
    T1b    0.112272
    T2     0.394256
    T3a    0.250653
    T3b    0.041775
    T4a    0.052219
    T4b    0.020888
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: N
    Absolute Frequencies:
    N0     268
    N1a     22
    N1b     93
    Name: count, dtype: int64
    
    Normalized Frequencies:
    N0     0.699739
    N1a    0.057441
    N1b    0.242820
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: M
    Absolute Frequencies:
    M0    365
    M1     18
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M0    0.953003
    M1    0.046997
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Stage
    Absolute Frequencies:
    I      333
    II      32
    III      4
    IVA      3
    IVB     11
    Name: count, dtype: int64
    
    Normalized Frequencies:
    I      0.869452
    II     0.083551
    III    0.010444
    IVA    0.007833
    IVB    0.028721
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Response
    Absolute Frequencies:
    Excellent                 208
    Structural Incomplete      91
    Biochemical Incomplete     23
    Indeterminate              61
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Excellent                 0.543081
    Structural Incomplete     0.237598
    Biochemical Incomplete    0.060052
    Indeterminate             0.159269
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Recurred
    Absolute Frequencies:
    No     275
    Yes    108
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.718016
    Yes    0.281984
    Name: proportion, dtype: float64
    --------------------------------------------------
    

## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. A total of 19 duplicated rows were identified.
    * In total, 34 observations were affected, consisting of 16 unique occurrences and 19 subsequent duplicates.
    * These 19 duplicates spanned 16 distinct variations, meaning some variations had multiple duplicates.
    * To clean the dataset, all 19 duplicate rows were removed, retaining only the first occurrence of each of the 16 unique variations.
2. No missing data noted for any variable with Null.Count>0 and Fill.Rate<1.0.
3. Low variance observed for 8 variables with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">Hx_Radiotherapy</span>: First.Second.Mode.Ratio = 51.000 (comprised 2 category levels)
    * <span style="color: #FF0000">M</span>: First.Second.Mode.Ratio = 19.222 (comprised 2 category levels)
    * <span style="color: #FF0000">Thyroid_Function</span>: First.Second.Mode.Ratio = 15.650 (comprised 5 category levels)
    * <span style="color: #FF0000">Hx_Smoking</span>: First.Second.Mode.Ratio = 12.000 (comprised 2 category levels)
    * <span style="color: #FF0000">Stage</span>: First.Second.Mode.Ratio = 9.812 (comprised 5 category levels)
    * <span style="color: #FF0000">Smoking</span>: First.Second.Mode.Ratio = 6.428 (comprised 2 category levels)
    * <span style="color: #FF0000">Pathology</span>: First.Second.Mode.Ratio = 6.022 (comprised 4 category levels)
    * <span style="color: #FF0000">Adenopathy</span>: First.Second.Mode.Ratio = 5.375 (comprised 5 category levels)
4. No low variance observed for any variable with Unique.Count.Ratio>10.
5. No high skewness observed for any variable with Skewness>3 or Skewness<(-3).



```python
##################################
# Counting the number of duplicated rows
##################################
thyroid_cancer.duplicated().sum()

```




    np.int64(19)




```python
##################################
# Exploring the duplicated rows
##################################
duplicated_rows = thyroid_cancer[thyroid_cancer.duplicated(keep=False)]
display(duplicated_rows)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>32</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>38</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>40</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>61</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>66</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>67</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>69</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>73</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>77</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>106</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>110</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>113</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>115</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>119</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>120</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>121</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>123</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>132</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>136</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>137</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>138</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>142</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>161</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>166</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>168</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>170</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>175</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>178</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>183</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>187</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>189</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
    <tr>
      <th>196</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Checking if duplicated rows have identical values across all columns
##################################
num_unique_dup_rows = duplicated_rows.drop_duplicates().shape[0]
num_total_dup_rows = duplicated_rows.shape[0]
if num_unique_dup_rows == 1:
    print("All duplicated rows have the same values across all columns.")
else:
    print(f"There are {num_unique_dup_rows} unique versions among the {num_total_dup_rows} duplicated rows.")
    
```

    There are 16 unique versions among the 35 duplicated rows.
    


```python
##################################
# Counting the unique variations among duplicated rows
##################################
unique_dup_variations = duplicated_rows.drop_duplicates()
variation_counts = duplicated_rows.value_counts().reset_index(name="Count")
print("Unique duplicated row variations and their counts:")
display(variation_counts)

```

    Unique duplicated row variations and their counts:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>29</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>36</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>37</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>38</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>40</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>42</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1b</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>51</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Removing the duplicated rows and
# retaining only the first occurrence
##################################
thyroid_cancer_row_filtered = thyroid_cancer.drop_duplicates(keep="first")
print('Dataset Dimensions: ')
display(thyroid_cancer_row_filtered.shape)

```

    Dataset Dimensions: 
    


    (364, 17)



```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(thyroid_cancer_row_filtered.dtypes)

```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(thyroid_cancer_row_filtered.columns)

```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(thyroid_cancer_row_filtered)] * len(thyroid_cancer_row_filtered.columns))

```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(thyroid_cancer_row_filtered.isna().sum(axis=0))

```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(thyroid_cancer_row_filtered.count())

```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)

```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>int64</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gender</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Smoking</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hx_Smoking</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hx_Radiotherapy</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Thyroid_Function</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Physical_Examination</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adenopathy</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pathology</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Focality</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Risk</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>T</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>N</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Stage</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Response</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Recurred</td>
      <td>category</td>
      <td>364</td>
      <td>364</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])

```




    0




```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]

```


```python
##################################
# Gathering the indices for each observation
##################################
row_index_list = thyroid_cancer_row_filtered.index

```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(thyroid_cancer_row_filtered.columns)] * len(thyroid_cancer_row_filtered))

```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(thyroid_cancer_row_filtered.isna().sum(axis=1))

```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)

```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_index_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>359</th>
      <td>378</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>360</th>
      <td>379</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>361</th>
      <td>380</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>362</th>
      <td>381</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>363</th>
      <td>382</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])

```




    0




```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
thyroid_cancer_numeric = thyroid_cancer_row_filtered.select_dtypes(include='number')

```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = thyroid_cancer_numeric.columns

```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = thyroid_cancer_numeric.min()

```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = thyroid_cancer_numeric.mean()

```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = thyroid_cancer_numeric.median()

```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = thyroid_cancer_numeric.max()

```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0] for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1] for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [thyroid_cancer_numeric[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [thyroid_cancer_numeric[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in thyroid_cancer_numeric]

```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)

```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = thyroid_cancer_numeric.nunique(dropna=True)

```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(thyroid_cancer_numeric)] * len(thyroid_cancer_numeric.columns))

```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)

```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = thyroid_cancer_numeric.skew()

```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = thyroid_cancer_numeric.kurtosis()

```


```python
##################################
# Generating a column quality summary for the numeric column
##################################
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Age</td>
      <td>15</td>
      <td>41.25</td>
      <td>38.0</td>
      <td>82</td>
      <td>31</td>
      <td>27</td>
      <td>21</td>
      <td>13</td>
      <td>1.615385</td>
      <td>65</td>
      <td>364</td>
      <td>0.178571</td>
      <td>0.678269</td>
      <td>-0.359255</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])

```




    0




```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])

```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])

```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
thyroid_cancer_categorical = thyroid_cancer_row_filtered.select_dtypes(include='category')

```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = thyroid_cancer_categorical.columns

```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [thyroid_cancer_row_filtered[x].value_counts().index.tolist()[0] for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [thyroid_cancer_row_filtered[x].value_counts().index.tolist()[1] for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [thyroid_cancer_categorical[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [thyroid_cancer_categorical[x].isin([thyroid_cancer_row_filtered[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in thyroid_cancer_categorical]

```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)

```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = thyroid_cancer_categorical.nunique(dropna=True)

```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(thyroid_cancer_categorical)] * len(thyroid_cancer_categorical.columns))

```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)

```


```python
##################################
# Generating a column quality summary for the categorical columns
##################################
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gender</td>
      <td>F</td>
      <td>M</td>
      <td>293</td>
      <td>71</td>
      <td>4.126761</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>315</td>
      <td>49</td>
      <td>6.428571</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hx_Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>336</td>
      <td>28</td>
      <td>12.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hx_Radiotherapy</td>
      <td>No</td>
      <td>Yes</td>
      <td>357</td>
      <td>7</td>
      <td>51.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thyroid_Function</td>
      <td>Euthyroid</td>
      <td>Clinical Hyperthyroidism</td>
      <td>313</td>
      <td>20</td>
      <td>15.650000</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Physical_Examination</td>
      <td>Multinodular goiter</td>
      <td>Single nodular goiter-right</td>
      <td>135</td>
      <td>127</td>
      <td>1.062992</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adenopathy</td>
      <td>No</td>
      <td>Right</td>
      <td>258</td>
      <td>48</td>
      <td>5.375000</td>
      <td>6</td>
      <td>364</td>
      <td>0.016484</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pathology</td>
      <td>Papillary</td>
      <td>Micropapillary</td>
      <td>271</td>
      <td>45</td>
      <td>6.022222</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Focality</td>
      <td>Uni-Focal</td>
      <td>Multi-Focal</td>
      <td>228</td>
      <td>136</td>
      <td>1.676471</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Risk</td>
      <td>Low</td>
      <td>Intermediate</td>
      <td>230</td>
      <td>102</td>
      <td>2.254902</td>
      <td>3</td>
      <td>364</td>
      <td>0.008242</td>
    </tr>
    <tr>
      <th>10</th>
      <td>T</td>
      <td>T2</td>
      <td>T3a</td>
      <td>138</td>
      <td>96</td>
      <td>1.437500</td>
      <td>7</td>
      <td>364</td>
      <td>0.019231</td>
    </tr>
    <tr>
      <th>11</th>
      <td>N</td>
      <td>N0</td>
      <td>N1b</td>
      <td>249</td>
      <td>93</td>
      <td>2.677419</td>
      <td>3</td>
      <td>364</td>
      <td>0.008242</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M</td>
      <td>M0</td>
      <td>M1</td>
      <td>346</td>
      <td>18</td>
      <td>19.222222</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stage</td>
      <td>I</td>
      <td>II</td>
      <td>314</td>
      <td>32</td>
      <td>9.812500</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Response</td>
      <td>Excellent</td>
      <td>Structural Incomplete</td>
      <td>189</td>
      <td>91</td>
      <td>2.076923</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Recurred</td>
      <td>No</td>
      <td>Yes</td>
      <td>256</td>
      <td>108</td>
      <td>2.370370</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])

```




    8




```python
##################################
# Identifying the categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Hx_Radiotherapy</td>
      <td>No</td>
      <td>Yes</td>
      <td>357</td>
      <td>7</td>
      <td>51.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M</td>
      <td>M0</td>
      <td>M1</td>
      <td>346</td>
      <td>18</td>
      <td>19.222222</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thyroid_Function</td>
      <td>Euthyroid</td>
      <td>Clinical Hyperthyroidism</td>
      <td>313</td>
      <td>20</td>
      <td>15.650000</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hx_Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>336</td>
      <td>28</td>
      <td>12.000000</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Stage</td>
      <td>I</td>
      <td>II</td>
      <td>314</td>
      <td>32</td>
      <td>9.812500</td>
      <td>5</td>
      <td>364</td>
      <td>0.013736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Smoking</td>
      <td>No</td>
      <td>Yes</td>
      <td>315</td>
      <td>49</td>
      <td>6.428571</td>
      <td>2</td>
      <td>364</td>
      <td>0.005495</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pathology</td>
      <td>Papillary</td>
      <td>Micropapillary</td>
      <td>271</td>
      <td>45</td>
      <td>6.022222</td>
      <td>4</td>
      <td>364</td>
      <td>0.010989</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adenopathy</td>
      <td>No</td>
      <td>Right</td>
      <td>258</td>
      <td>48</td>
      <td>5.375000</td>
      <td>6</td>
      <td>364</td>
      <td>0.016484</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])

```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>

### 1.4.1 Ordinal Binning <a class="anchor" id="1.4.1"></a>


```python
##################################
# Creating a dataset copy
# of the row filtered data
##################################
thyroid_cancer_baseline = thyroid_cancer_row_filtered.copy()
```


```python
##################################
# Defining bins and labels
##################################
bins = [0, 50, float('inf')]
labels = ['<50', '50+']
```


```python
##################################
# Creating ordinal bins
# for the numeric column
##################################
thyroid_cancer_baseline['Age_Group'] = pd.cut(thyroid_cancer_baseline['Age'], bins=bins, labels=labels, right=False)
thyroid_cancer_baseline['Age_Group'] = pd.Categorical(thyroid_cancer_baseline['Age_Group'], categories=labels, ordered=True)
display(thyroid_cancer_baseline)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Recurred</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate</td>
      <td>No</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>No</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>378</th>
      <td>72</td>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>Right</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Biochemical Incomplete</td>
      <td>Yes</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>379</th>
      <td>81</td>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Extensive</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>380</th>
      <td>72</td>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Bilateral</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>381</th>
      <td>61</td>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Clinical Hyperthyroidism</td>
      <td>Multinodular goiter</td>
      <td>Extensive</td>
      <td>Hurthle Cell</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M0</td>
      <td>IVA</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>382</th>
      <td>67</td>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Bilateral</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M0</td>
      <td>IVA</td>
      <td>Structural Incomplete</td>
      <td>Yes</td>
      <td>50+</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 18 columns</p>
</div>



```python
##################################
# Preparing the working dataset
# by excluding unnecessary columns
##################################
exclude_cols = ['Age', 'Recurred']
thyroid_cancer_baseline_filtered = thyroid_cancer_baseline.drop(columns=exclude_cols)
display(thyroid_cancer_baseline_filtered)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-left</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>No</td>
      <td>Micropapillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1a</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>378</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Single nodular goiter-right</td>
      <td>Right</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Biochemical Incomplete</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>379</th>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Extensive</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Structural Incomplete</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>380</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Bilateral</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M1</td>
      <td>IVB</td>
      <td>Structural Incomplete</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>381</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Clinical Hyperthyroidism</td>
      <td>Multinodular goiter</td>
      <td>Extensive</td>
      <td>Hurthle Cell</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M0</td>
      <td>IVA</td>
      <td>Structural Incomplete</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>382</th>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular goiter</td>
      <td>Bilateral</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>High</td>
      <td>T4b</td>
      <td>N1b</td>
      <td>M0</td>
      <td>IVA</td>
      <td>Structural Incomplete</td>
      <td>50+</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 16 columns</p>
</div>


### 1.4.2 Category Aggregration and Encoding <a class="anchor" id="1.4.2"></a>


```python
##################################
# Merging small categories into broader groups 
# for certain categorical predictors
# to ensure sufficient representation in statistical models 
# and prevent sparsity issues in cross-validation
##################################
thyroid_cancer_baseline_filtered['Thyroid_Function'] = thyroid_cancer_baseline_filtered['Thyroid_Function'].map(lambda x: 'Euthyroid' if (x in ['Euthyroid'])  else 'Hypothyroidism or Hyperthyroidism').astype('category')
thyroid_cancer_baseline_filtered['Physical_Examination'] = thyroid_cancer_baseline_filtered['Physical_Examination'].map(lambda x: 'Normal or Single Nodular Goiter' if (x in ['Normal', 'Single nodular goiter-left', 'Single nodular goiter-right'])  else 'Multinodular or Diffuse Goiter').astype('category')
thyroid_cancer_baseline_filtered['Adenopathy'] = thyroid_cancer_baseline_filtered['Adenopathy'].map(lambda x: 'No' if x == 'No' else ('Yes' if pd.notna(x) and x != '' else x)).astype('category')
thyroid_cancer_baseline_filtered['Pathology'] = thyroid_cancer_baseline_filtered['Pathology'].map(lambda x: 'Non-Papillary' if (x in ['Hurthle Cell', 'Follicular'])  else 'Papillary').astype('category')
thyroid_cancer_baseline_filtered['Risk'] = thyroid_cancer_baseline_filtered['Risk'].map(lambda x: 'Low' if (x in ['Low'])  else 'Intermediate to High').astype('category')
thyroid_cancer_baseline_filtered['T'] = thyroid_cancer_baseline_filtered['T'].map(lambda x: 'T1 to T2' if (x in ['T1a', 'T1b', 'T2'])  else 'T3 to T4b').astype('category')
thyroid_cancer_baseline_filtered['N'] = thyroid_cancer_baseline_filtered['N'].map(lambda x: 'N0' if (x in ['N0'])  else 'N1').astype('category')
thyroid_cancer_baseline_filtered['Stage'] = thyroid_cancer_baseline_filtered['Stage'].map(lambda x: 'I' if (x in ['I'])  else 'II to IVB').astype('category')
thyroid_cancer_baseline_filtered['Response'] = thyroid_cancer_baseline_filtered['Response'].map(lambda x: 'Indeterminate or Incomplete' if (x in ['Indeterminate', 'Structural Incomplete', 'Biochemical Incomplete'])  else 'Excellent').astype('category')
thyroid_cancer_baseline_filtered.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate or Incomplete</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4.3 Synthetic Outlier Labeling via Frequency-Based Tagging <a class="anchor" id="1.4.3"></a>


```python
##################################
# Defining a function for implementing a
# frequency-based outlier tagging
##################################
def frequency_based_outlier_tagging(df, threshold=0.005):
    freq = df.value_counts(normalize=True)
    rare_patterns = freq[freq < threshold].index
    outlier_mask = df.apply(lambda row: tuple(row) in rare_patterns, axis=1)
    labels = ['No', 'Yes']
    return pd.Categorical(outlier_mask.map({True: 'Yes', False: 'No'}), categories=labels, ordered=True)
thyroid_cancer_baseline_filtered['Outlier'] = frequency_based_outlier_tagging(thyroid_cancer_baseline_filtered.drop(columns='Outlier', errors='ignore'))
display(thyroid_cancer_baseline_filtered)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Smoking</th>
      <th>Hx_Smoking</th>
      <th>Hx_Radiotherapy</th>
      <th>Thyroid_Function</th>
      <th>Physical_Examination</th>
      <th>Adenopathy</th>
      <th>Pathology</th>
      <th>Focality</th>
      <th>Risk</th>
      <th>T</th>
      <th>N</th>
      <th>M</th>
      <th>Stage</th>
      <th>Response</th>
      <th>Age_Group</th>
      <th>Outlier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Indeterminate or Incomplete</td>
      <td>&lt;50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>&lt;50</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>F</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>No</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Low</td>
      <td>T1 to T2</td>
      <td>N0</td>
      <td>M0</td>
      <td>I</td>
      <td>Excellent</td>
      <td>50+</td>
      <td>No</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>378</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Normal or Single Nodular Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Uni-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>M1</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>50+</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>379</th>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>M1</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>50+</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>380</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>M1</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>50+</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>381</th>
      <td>M</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Hypothyroidism or Hyperthyroidism</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Non-Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>M0</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>50+</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>382</th>
      <td>M</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Euthyroid</td>
      <td>Multinodular or Diffuse Goiter</td>
      <td>Yes</td>
      <td>Papillary</td>
      <td>Multi-Focal</td>
      <td>Intermediate to High</td>
      <td>T3 to T4b</td>
      <td>N1</td>
      <td>M0</td>
      <td>II to IVB</td>
      <td>Indeterminate or Incomplete</td>
      <td>50+</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 17 columns</p>
</div>



```python
##################################
# Performing a general exploration of the categorical variable levels
# based on the ordered categories
##################################
ordered_cat_cols = thyroid_cancer_baseline_filtered.columns
for col in ordered_cat_cols:
    print(f"Column: {col}")
    print("Absolute Frequencies:")
    print(thyroid_cancer_baseline_filtered[col].value_counts().reindex(thyroid_cancer_baseline_filtered[col].cat.categories))
    print("\nNormalized Frequencies:")
    print(thyroid_cancer_baseline_filtered[col].value_counts(normalize=True).reindex(thyroid_cancer_baseline_filtered[col].cat.categories))
    print("-" * 50)
    
```

    Column: Gender
    Absolute Frequencies:
    M     71
    F    293
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M    0.195055
    F    0.804945
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Smoking
    Absolute Frequencies:
    No     315
    Yes     49
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.865385
    Yes    0.134615
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Smoking
    Absolute Frequencies:
    No     336
    Yes     28
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.923077
    Yes    0.076923
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Hx_Radiotherapy
    Absolute Frequencies:
    No     357
    Yes      7
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.980769
    Yes    0.019231
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Thyroid_Function
    Absolute Frequencies:
    Euthyroid                            313
    Hypothyroidism or Hyperthyroidism     51
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Euthyroid                            0.85989
    Hypothyroidism or Hyperthyroidism    0.14011
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Physical_Examination
    Absolute Frequencies:
    Multinodular or Diffuse Goiter     142
    Normal or Single Nodular Goiter    222
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Multinodular or Diffuse Goiter     0.39011
    Normal or Single Nodular Goiter    0.60989
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Adenopathy
    Absolute Frequencies:
    No     258
    Yes    106
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.708791
    Yes    0.291209
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Pathology
    Absolute Frequencies:
    Non-Papillary     48
    Papillary        316
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Non-Papillary    0.131868
    Papillary        0.868132
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Focality
    Absolute Frequencies:
    Uni-Focal      228
    Multi-Focal    136
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Uni-Focal      0.626374
    Multi-Focal    0.373626
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Risk
    Absolute Frequencies:
    Intermediate to High    134
    Low                     230
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Intermediate to High    0.368132
    Low                     0.631868
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: T
    Absolute Frequencies:
    T1 to T2     224
    T3 to T4b    140
    Name: count, dtype: int64
    
    Normalized Frequencies:
    T1 to T2     0.615385
    T3 to T4b    0.384615
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: N
    Absolute Frequencies:
    N0    249
    N1    115
    Name: count, dtype: int64
    
    Normalized Frequencies:
    N0    0.684066
    N1    0.315934
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: M
    Absolute Frequencies:
    M0    346
    M1     18
    Name: count, dtype: int64
    
    Normalized Frequencies:
    M0    0.950549
    M1    0.049451
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Stage
    Absolute Frequencies:
    I            314
    II to IVB     50
    Name: count, dtype: int64
    
    Normalized Frequencies:
    I            0.862637
    II to IVB    0.137363
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Response
    Absolute Frequencies:
    Excellent                      189
    Indeterminate or Incomplete    175
    Name: count, dtype: int64
    
    Normalized Frequencies:
    Excellent                      0.519231
    Indeterminate or Incomplete    0.480769
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Age_Group
    Absolute Frequencies:
    <50    258
    50+    106
    Name: count, dtype: int64
    
    Normalized Frequencies:
    <50    0.708791
    50+    0.291209
    Name: proportion, dtype: float64
    --------------------------------------------------
    Column: Outlier
    Absolute Frequencies:
    No     220
    Yes    144
    Name: count, dtype: int64
    
    Normalized Frequencies:
    No     0.604396
    Yes    0.395604
    Name: proportion, dtype: float64
    --------------------------------------------------
    

### 1.4.4 Data Splitting <a class="anchor" id="1.4.4"></a>


```python
##################################
# Creating a dataset copy
# of the aggregated and encoded data
##################################
thyroid_cancer_presplitting = thyroid_cancer_baseline_filtered.copy()

```


```python
##################################
# Performing a general exploration
# of the presplitting dataset
##################################
print('Final Dataset Dimensions: ')
display(thyroid_cancer_presplitting.shape)

```

    Final Dataset Dimensions: 
    


    (364, 17)



```python
##################################
# Exploring the outlier breakdown
##################################
print('Target Variable Breakdown: ')
thyroid_cancer_breakdown = thyroid_cancer_presplitting.groupby('Outlier', observed=True).size().reset_index(name='Count')
thyroid_cancer_breakdown['Percentage'] = (thyroid_cancer_breakdown['Count'] / len(thyroid_cancer_baseline)) * 100
display(thyroid_cancer_breakdown)

```

    Target Variable Breakdown: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Outlier</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>No</td>
      <td>220</td>
      <td>60.43956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Yes</td>
      <td>144</td>
      <td>39.56044</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the train and test data
# from the final dataset
# by applying stratification and
# using a 75-25 ratio
##################################
thyroid_cancer_train_initial, thyroid_cancer_test = train_test_split(thyroid_cancer_presplitting, 
                                                                     test_size=0.25, 
                                                                     stratify=thyroid_cancer_presplitting['Outlier'], 
                                                                     random_state=987654321)

```


```python
##################################
# Performing a general exploration
# of the initial training dataset
##################################
X_train_initial = thyroid_cancer_train_initial.drop('Outlier', axis = 1)
y_train_initial = thyroid_cancer_train_initial['Outlier']
print('Initial Train Dataset Dimensions: ')
display(X_train_initial.shape)
display(y_train_initial.shape)
print('Initial Train Target Variable Breakdown: ')
display(y_train_initial.value_counts())
print('Initial Train Target Variable Proportion: ')
display(y_train_initial.value_counts(normalize = True))

```

    Initial Train Dataset Dimensions: 
    


    (273, 16)



    (273,)


    Initial Train Target Variable Breakdown: 
    


    Outlier
    No     165
    Yes    108
    Name: count, dtype: int64


    Initial Train Target Variable Proportion: 
    


    Outlier
    No     0.604396
    Yes    0.395604
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the test dataset
##################################
X_test = thyroid_cancer_test.drop('Outlier', axis = 1)
y_test = thyroid_cancer_test['Outlier']
print('Test Dataset Dimensions: ')
display(X_test.shape)
display(y_test.shape)
print('Test Target Variable Breakdown: ')
display(y_test.value_counts())
print('Test Target Variable Proportion: ')
display(y_test.value_counts(normalize = True))

```

    Test Dataset Dimensions: 
    


    (91, 16)



    (91,)


    Test Target Variable Breakdown: 
    


    Outlier
    No     55
    Yes    36
    Name: count, dtype: int64


    Test Target Variable Proportion: 
    


    Outlier
    No     0.604396
    Yes    0.395604
    Name: proportion, dtype: float64



```python
##################################
# Formulating the train and validation data
# from the train dataset
# by applying stratification and
# using a 75-25 ratio
##################################
thyroid_cancer_train, thyroid_cancer_validation = train_test_split(thyroid_cancer_train_initial, 
                                                                   test_size=0.25, 
                                                                   stratify=thyroid_cancer_train_initial['Outlier'], 
                                                                   random_state=987654321)

```


```python
##################################
# Performing a general exploration
# of the final training dataset
##################################
X_train = thyroid_cancer_train.drop('Outlier', axis = 1)
y_train = thyroid_cancer_train['Outlier']
print('Final Train Dataset Dimensions: ')
display(X_train.shape)
display(y_train.shape)
print('Final Train Target Variable Breakdown: ')
display(y_train.value_counts())
print('Final Train Target Variable Proportion: ')
display(y_train.value_counts(normalize = True))

```

    Final Train Dataset Dimensions: 
    


    (204, 16)



    (204,)


    Final Train Target Variable Breakdown: 
    


    Outlier
    No     123
    Yes     81
    Name: count, dtype: int64


    Final Train Target Variable Proportion: 
    


    Outlier
    No     0.602941
    Yes    0.397059
    Name: proportion, dtype: float64



```python
##################################
# Performing a general exploration
# of the validation dataset
##################################
X_validation = thyroid_cancer_validation.drop('Outlier', axis = 1)
y_validation = thyroid_cancer_validation['Outlier']
print('Validation Dataset Dimensions: ')
display(X_validation.shape)
display(y_validation.shape)
print('Validation Target Variable Breakdown: ')
display(y_validation.value_counts())
print('Validation Target Variable Proportion: ')
display(y_validation.value_counts(normalize = True))

```

    Validation Dataset Dimensions: 
    


    (69, 16)



    (69,)


    Validation Target Variable Breakdown: 
    


    Outlier
    No     42
    Yes    27
    Name: count, dtype: int64


    Validation Target Variable Proportion: 
    


    Outlier
    No     0.608696
    Yes    0.391304
    Name: proportion, dtype: float64



```python
##################################
# Saving the training data
# to the DATASETS_FINAL_TRAIN_PATH
# and DATASETS_FINAL_TRAIN_FEATURES_PATH
# and DATASETS_FINAL_TRAIN_TARGET_PATH
##################################
thyroid_cancer_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_PATH, "thyroid_cancer_train.csv"), index=False)
X_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_FEATURES_PATH, "X_train.csv"), index=False)
y_train.to_csv(os.path.join("..", DATASETS_FINAL_TRAIN_TARGET_PATH, "y_train.csv"), index=False)

```


```python
##################################
# Saving the validation data
# to the DATASETS_FINAL_VALIDATION_PATH
# and DATASETS_FINAL_VALIDATION_FEATURE_PATH
# and DATASETS_FINAL_VALIDATION_TARGET_PATH
##################################
thyroid_cancer_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_PATH, "thyroid_cancer_validation.csv"), index=False)
X_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_FEATURES_PATH, "X_validation.csv"), index=False)
y_validation.to_csv(os.path.join("..", DATASETS_FINAL_VALIDATION_TARGET_PATH, "y_validation.csv"), index=False)

```


```python
##################################
# Saving the test data
# to the DATASETS_FINAL_TEST_PATH
# and DATASETS_FINAL_TEST_FEATURES_PATH
# and DATASETS_FINAL_TEST_TARGET_PATH
##################################
thyroid_cancer_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_PATH, "thyroid_cancer_test.csv"), index=False)
X_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_FEATURES_PATH, "X_test.csv"), index=False)
y_test.to_csv(os.path.join("..", DATASETS_FINAL_TEST_TARGET_PATH, "y_test.csv"), index=False)

```

### 1.4.5 Data Preparation <a class="anchor" id="1.4.5"></a>

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

