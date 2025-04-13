***
# Supervised | Unsupervised Learning : Detecting and Evaluating Anomalies in Categorical Data Under Supervised and Unsupervised Settings

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *April 19, 2025*
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

