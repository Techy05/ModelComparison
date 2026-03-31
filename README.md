# Set-up

- Set up python virtual environment
~~~
python -m venv venv
.\venv\Scripts\Activate.ps1
~~~
- Install the required libraries
~~~
pip install pandas scikit-learn
~~~
- Run main.py using VSCode or
~~~
python -m main.py
~~~

# Approach
```sklearn``` library contains pre-built functions to train, test, and calculate the accuracy of each model. 
- ```LogisticRegression```
- ```KNeighborsClassifier```
- ```DecisionTreeClassifier```
<br>

# Data Set Used

## IRIS
One of the earliest known datasets used for evaluating classification methods. As mentioned on official website:
> This is one of the earliest datasets used in the literature on classification methods and widely used in statistics and machine learning.  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is linearly separable from the other 2; the latter are not linearly separable from each other.
Predicted attribute: class of iris plant.
<br>


# Model comparision

## Logistic Regression
Logistic Regression is used for binary classification. It follows an S-shaped curve which predicts the probability of a certain class, instead of continuous values.

In simple words, instead of predicting what type of class a plant is, it predicts the probability of this plant belonging to this class, for all classes seperately, and then picks the one with highest probability.

## K-Nearest Neighbours
KNN works by measuring the distance between a new data point and existing labeled points, assigning the new point to the majority class among its
closest neighbors.

It doesn't build a model during training. When you give it a test sample, it finds the K most similar samples from training data and takes a majority vote. Its weakness is that it slows down significantly on large datasets because it has to calculate distance to every training point at prediction/testing time.

## Decision Tree
Decision Tree builds a flowchart of yes/no decisions by repeatedly asking certain questions to split the training data as much as possible, and then follows the tree/model to classify test data.

It's the most human-readable of the three. But, Decision Trees have a tendency to overfit. Without any constraints, the tree will keep splitting until it has perfectly memorised every training sample, which often makes it perform worse on the test set compared to the other two models. 

<br>


# Accuracy
The data set was divided into 80% training and 20% testing set.

The attribute ```random_state``` was set to ```95``` to keep the data distribution exactly same across all runs and models.
95 is just a random number. Can be set to anything, which I have done below.

|      MODEL NAME      | ACCURACY |
| :------------------: | -------: |
| Logistic Regression  |   0.967  |
| KNN                  |   0.967  |
| Decision Tree        |   0.933   |

Now setting ```random_state=6969```

|      MODEL NAME      | ACCURACY |
| :------------------: | -------: |
| Logistic Regression  |   1.000  |
| KNN                  |   1.000  |
| Decision Tree        |   0.967  |

```random_state=23456```

|      MODEL NAME      | ACCURACY |
| :------------------: | -------: |
| Logistic Regression  |   0.933  |
| KNN                  |   0.933  |
| Decision Tree        |   0.933   |

I tried with 10 different values of ```random_state``` to split the data differently:
- in 9 of them, LR performed better or as good as others.
- in 6 of them, KNN performed better or as good as others.
- in 6 of them, DT performed better or as good as others.

### WINNER: <ins>Logistic Regression</ins>
### Why is the difference so low though?
- To be clear, this is not enough testing to prove which model is universally better. 

- In IRIS dataset, one class is linearly separable from the other 2, the latter are not linearly separable from each other.
- Logistic Regression performs the best when all classes are seperable from each other. The overlapping data may have gone into training data while splitting, more times than not. That boosted the accuracy of LR over the other 2 models. But whenever, that overlapping made its way into testing data, LR fell behind or as low as KNN and Decision Tree.
- To get fair results, instead of splitting the data manually using ```random_state```, **cross-validation** is used, which splits the data in multiple different ways and selects different testing data each time. Then average of all results is taken. In that scenario, KNN and Decision Tree can match LR in IRIS dataset.

