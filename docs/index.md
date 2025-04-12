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
