# Capstone-Project-part3
Advanced Modeling — Ensembles, Tuning, and Full ML Pipeline 

1.Baseline & Overfitting Analysis

Before building robust ensemble models, we established a baseline using a single, unconstrained `DecisionTreeClassifier` (leaving `max_depth` at its default `None`). 

Baseline Performance
* **Training Accuracy:** 100.0%
* **Test Accuracy:** 57.0%

Evidence of Overfitting and High Variance
This baseline model exhibits textbook signs of severe **overfitting**: it perfectly memorised the training data (achieving 100% accuracy) but generalized poorly to the unseen test data (dropping to just 57% accuracy). 

**Why does this happen?**
Single decision trees are notoriously described as **high-variance models**. When left unconstrained, a decision tree will continue splitting the data until every single leaf node is "pure" (containing only one class). It does this greedily: at each step, it makes the optimal split for the data immediately in front of it without revisiting earlier decisions or considering the global structure. Consequently, the tree grows highly complex and ends up modeling the random noise and unique idiosyncrasies of the training set rather than learning the underlying, generalizable patterns of customer churn. This high sensitivity (variance) to the specific training data is exactly what ensemble methods (like Random Forests) are designed to solve.

2.Controlled Decision Tree Evaluation
To address the severe overfitting observed in our baseline model, we trained a second DecisionTreeClassifier with specific hyperparameter constraints designed to limit the model's complexity.

Performance Results:

Training Accuracy: 100.0%

Testing Accuracy: 57.0%

Understanding the Hyperparameters:
By default, a decision tree will grow greedily until it memorizes the training data. We introduced two constraints (regularization techniques) to force the model to learn more generalized patterns:

max_depth=5: This parameter acts as a hard limit on how many levels deep the tree can grow. By capping the depth, we stop the model from creating highly complex, convoluted rules that only apply to a few isolated data points. This constraint intentionally introduces a slight amount of bias (simplifying the model) in exchange for a massive reduction in variance (preventing overfitting).

min_samples_split=20: This parameter dictates that an internal node must contain at least 20 customer records before it is allowed to split again. If a node has 19 or fewer samples, it becomes a leaf. This prevents the tree from creating highly specific branches that merely respond to the random statistical noise present in tiny data subsets.

Train/Test Gap Comparison:
The impact of these constraints is clearly visible when comparing the train/test performance gaps of the two models.

Unconstrained Baseline Gap: 43.0% (100.0% Train vs. 57.0% Test)

Controlled Tree Gap:0.75%(73.25% Train vs. 72.50%)

While the controlled tree's training accuracy dropped compared to the baseline, its testing accuracy significantly improved. The drastically narrower gap between training and testing performance indicates that the model is no longer memorizing the data. Instead, it has successfully learned underlying, generalizable trends about customer churn that hold true even when evaluating unseen data.

3.Evaluating Split Criteria: Gini Impurity vs. EntropyTo determine the optimal way to split our data at each node, a Decision Tree evaluates different features using a mathematical loss function. The goal is always to maximize the "purity" of the resulting child nodes. We tested the two most common impurity metrics: Gini and Entropy.

Performance Comparison:
Gini Test Accuracy:72.00%
 
Entropy Test Accuracy: 72.50%

Note: In practice, Gini and Entropy yield remarkably similar tree structures and performance metrics. Gini is often preferred as the default because it is slightly faster to compute computationally (as it avoids logarithmic functions).The Mathematical DefinitionsIn both formulas below, $p_i$ represents the probability (or fraction) of items belonging to class $i$ for a given node.

 Gini Impurity :Gini impurity measures the probability of incorrectly classifying a randomly chosen element in the node if it were randomly labeled according to the distribution of labels in the node.

    {Gini} = 1 - \sum_{i=1}^{C} p_i^2
  What does a Gini of 0 mean?A Gini score of 0 represents absolute certainty, or a completely "pure" node. It means that every single customer sample in that specific node belongs to the exact same class (e.g., 100% of the samples in the node are "Churn = Yes"). When a node hits a Gini of 0, the tree stops splitting that branch.

 Entropy :Rooted in information theory, entropy measures the level of disorder or uncertainty within a node.

    {Entropy} = -\sum_{i=1}^{C} p_i \log_2(p_i)
   Similar to Gini, an Entropy of 0 indicates a perfectly pure node with zero disorder. The maximum entropy (for a binary classification task like ours) is 1, which occurs when the node is perfectly split 50/50 between churners and non-churners, representing maximum uncertainty.

4.Feature Importance: Random Forest vs. Linear Regression

In a Random Forest, feature importance is calculated as the average reduction in Gini impurity across all trees in the ensemble. Every time a specific feature is used to split a node, the model calculates how much that split decreased the impurity (weighted by the number of samples passing through that node). The total decrease is averaged across all 100 trees to yield the final importance score.This differs fundamentally from a linear regression coefficient. A regression coefficient represents the specific magnitude and direction (positive or negative) of the linear relationship between a feature and the target, assuming all other features are held constant. In contrast, Random Forest feature importance does not indicate directionality or assume a linear relationship; it purely measures how useful a feature is for effectively separating the classes, even in highly non-linear scenarios or complex interactions with other variables.

Bagging and Variance Reduction

Random Forests utilize an ensemble method called Bagging (Bootstrap Aggregating) to drastically improve generalization. First, it uses bootstrap sampling, meaning each of the 100 decision trees is trained on a random sample of the training data drawn with replacement—so some customer records are repeated while others are left out entirely. Furthermore, at every single node split, the tree is only allowed to consider a random subset of features(typically $\sqrt{\text{total features}}$). This forced randomization ensures that the trees within the ensemble are highly diverse and un-correlated. While a single deep decision tree is highly sensitive to statistical noise (high variance) and tends to overfit, averaging the independent predictions of hundreds of these diverse trees smooths out the idiosyncratic errors, resulting in a highly robust model with drastically reduced variance.


Feature Ablation Study: Impact of Dimensionality Reduction

To evaluate the true value of our model's features, we conducted an ablation study by completely removing the 5 features with the lowest importance scores and retraining an identical Random Forest classifier.

Observation on Feature Informativeness:

Full Model ROC-AUC: 0.6712

Reduced Model ROC-AUC: 0.6720

AUC Improvement: The testing metrics revealed that the removed features were genuinely uninformative. Stripping them out did not degrade the model's predictive power, and may have even slightly improved it by removing statistical noise that the model was previously trying to parse.

Implications for Production Deployment:
This ablation study highlights a critical concept in Machine Learning engineering: the trade-off between model complexity and production efficiency. Deploying a simpler, lower-dimensional model (one that requires fewer features) offers significant engineering advantages:

Lower Inference Latency: Models with fewer features process data faster at runtime, which is crucial for real-time applications.

Reduced Pipeline Complexity: Every feature a model requires means a data pipeline must be built, monitored, and maintained to supply it. Removing 5 features eliminates 5 potential points of failure in production data pipelines.

Decreased Compute/Storage Costs: Processing a narrower matrix requires fewer compute resources and memory.

Ultimately, intentionally sacrificing a negligible amount of predictive accuracy is often highly acceptable—and even preferred—if it drastically reduces the engineering overhead, maintenance burden, and latency of deploying the model into a live production environment.

Cross Validation Comparison

A single train-test split evaluates a model's performance on one specific, randomized division of the data. This creates a vulnerability: the model's performance score is highly dependent on which exact customer records randomly ended up in the test set. By sheer chance, the test set might contain disproportionately "easy" or "difficult" records to classify, leading to an overly optimistic or overly pessimistic evaluation of the model's true capability.

Cross-validation (specifically 5-fold Stratified K-Fold) solves this by dividing the entire training dataset into five distinct partitions (folds), ensuring the class balance is maintained in each. The model is then trained and evaluated five separate times. In each iteration, a different fold is held out as the test set, while the remaining four folds are used for training.

This methodology offers two distinct advantages for estimating generalization performance:

Comprehensive Evaluation: Every single data point in the dataset gets to act as testing data exactly once. This ensures the evaluation covers the full variance and complexity of the dataset.

Measurement of Stability: By outputting a mean score and a standard deviation, cross-validation tells us not only how well the model performs on average, but how stable that performance is. A high standard deviation across the folds indicates that the model is highly sensitive to changes in the training data, warning us of potential brittleness when deployed to production.


Hyperparameter Tuning: GridSearchCV & Model Selection
To optimize the Random Forest model and find the best balance between bias and variance, we performed an exhaustive hyperparameter search using GridSearchCV. We wrapped the imputer, scaler, and classifier in a Scikit-Learn Pipeline to prevent data leakage during the cross-validation process.

Total Configurations Evaluated
The parameter grid tested multiple values across three key hyperparameters:

n_estimators: [50, 100, 200] (3 options)

max_depth: [5, 10, None] (3 options)

min_samples_leaf: [1, 5] (2 options)

This resulted in 18 distinct model configurations (3 × 3 × 2). Because we utilized Stratified 5-Fold Cross-Validation, the grid search trained and evaluated a total of 90 model fits (18 combinations × 5 folds).

Optimized Results & Insights
The grid search achieved a best cross-validated ROC-AUC Score of 0.7136.

The algorithm selected the following as the optimal hyperparameters:

max_depth: 5. The selection of a heavily constrained depth confirms our earlier hypothesis: this dataset is highly prone to overfitting. Limiting the tree depth is strictly necessary to force the model to learn generalizable patterns rather than memorizing statistical noise.

min_samples_leaf: 5. By forcing each leaf to contain at least 5 samples, the model is further protected against creating highly specific, convoluted rules for isolated data points.

n_estimators: 200. The model preferred the maximum number of trees offered in the grid. Increasing the number of trees in a Random Forest leverages the power of ensemble bagging, which drives down variance and improves model stability without increasing the risk of overfitting.

The Trade-off: Exhaustive Grid Search vs. Randomized Search
Hyperparameter tuning presents a strict trade-off between computational cost and finding the mathematical optimum.

We utilized an Exhaustive Grid Search (GridSearchCV). This method systematically brute-forces every single possible combination within the defined parameter grid. While this guarantees finding the absolute best configuration within that specific space, it suffers from the "curse of dimensionality." As the number of hyperparameters and data volume grows, the computational cost grows exponentially.

In larger-scale machine learning projects, engineers typically rely on a Randomized Search (RandomizedSearchCV) or Bayesian optimization. Instead of trying every combination, randomized search samples a fixed number of parameter settings from specified statistical distributions. While it does not guarantee finding the absolute mathematical optimum, it explores a wider variety of continuous values and typically finds a near-optimal solution using a fraction of the computational time, memory, and processing power. Because our parameter grid was relatively small (18 combinations), the exhaustive Grid Search was highly practical and yielded a stable, production-ready model.

Learning Curve Analysis: Data Quantity vs. Model Capacity
To determine if our model would benefit from more data or a more complex architecture, we plotted a manual learning curve by retraining our optimized pipeline on progressively larger subsets of the training data (20%, 40%, 60%, 80%, and 100%).

(i) Training AUC Behavior: As expected for a model with sufficient capacity, the Training AUC decreases slightly as the training set grows. When trained on only 20% of the data, the model can easily memorize the small number of examples, resulting in an artificially high training score. As more data is introduced, the complexity and variance of the dataset increase, making it harder for the model to perfectly fit every data point.

(ii) Test AUC Behavior: Conversely, the Test AUC increases as more training data is added. Exposing the model to a larger, more diverse set of customer records allows it to learn robust, generalizable patterns rather than fixating on the statistical noise of a small subset. This upward trajectory confirms that collecting data improves predictive performance.

(iii) Conclusion on Current Limitations:
  Looking at the final iterations, the Test AUC is still clearly rising at 100% of the data. This indicates that our model is currently limited by data quantity. Investing resources into acquiring more customer records would likely continue to yield improvements in predictive accuracy.

Final Comparison Table and Recommendation

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |

| **Logistic Regression** | 0.7053 | 0.0197 | 0.7133 |

| **Decision Tree** (max_depth=5) | 0.7154 | 0.0193 | 0.6943 |

| **Random Forest** (Base) | 0.6964 | 0.0313 | 0.6712 |

| **Gradient Boosting** | 0.7179 | 0.0304 | 0.7097 |

| **Random Forest** (GridSearchCV Tuned) | 0.7136 | 0.0290 | 0.7087 |

Client Recommendation
I highly recommend deploying the Logistic Regression model for the client's churn prediction needs. While the Gradient Boosting model achieved a marginally higher mean score during cross-validation, the Logistic Regression model demonstrated superior stability with the lowest standard deviation across folds (0.0197) and ultimately achieved the highest ROC-AUC on the unseen test set (0.7133). Furthermore, as a linear model, it offers excellent interpretability, allowing you to clearly explain the exact driving factors and coefficient weights behind customer churn to non-technical business stakeholders without the "black box" complexity of ensemble methods.









