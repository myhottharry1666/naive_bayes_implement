# Multinomial Naive Bayes Classifier Implementation

In this project, we implement the Multinomial Naive Bayes Classifier (MultinomialNB).  
We test the implemented MultinomialNB on fetch_20newsgroups dataset from sklearn, comparing the accuracy with MultinomialNB implemented by sklearn.

## Detailed project description

### Files
"naive_bayes_by_myself.py" implements MultinomialNB from scratch.  
"test_naive_bayes.py" does all the tests of MultinomialNB through fetch_20newsgroups dataset from sklearn.

**jupyter notebook**  
"naive_bayes_sklearn_demo.ipynb" stores a demo, training MultinomialNB through sklearn built-in class, and calculating accuracy through fetch_20newsgroups dataset.  
"naive_bayes_myself_demo.ipynb" stores a demo, training MultinomialNB through my implemented class, and calculating accuracy through fetch_20newsgroups dataset.


### Tests
To test my implemented MultinomialNB, in "test_naive_bayes.py", we build a MultinomialNB model by sklearn built-in class, and another model by my implemented class, both through fetch_20newsgroups dataset from sklearn.  
Then we will compare the accuracy of two models, calculated by predicting test dataset. We also run two models for multiple times, plotting their accuracy for comparison.

Note: My MultinomialNB is a bit slow regarding runtime, so feel free to adjust
the number of times to run in main() function (run on the whole test dataset for one time takes around 2-3 minutes).

## Instructions to use my codes

Please follow steps listed below one by one, from top to bottom.

### Put python files

Put all my python files ("naive_bayes_by_myself.py", "test_naive_bayes.py") in the same folder, say folderX.
If you want to run those two jupyter notebooks, also put "naive_bayes_sklearn_demo.ipynb" and "naive_bayes_myself_demo.ipynb" in folder X.

### Required libraries

Open your terminal, use pip to install these libraries:

 1. scikit-learn
 2. numpy
 3. matplotlib

##  Overview of performance
Regarding accuracy on test dataset, compared with sklearn built-in MultinomialNB, my implemented MultinomialNB will have ±1%-5% difference.  
I run both MultinomialNB for five times, and plot their accuracy in the chart below.
![avatar](https://drive.google.com/uc?export=view&id=1bwjIVfGaIiKWsiajDqka_aRkLNZNamGK)
In summary, we can see that accuracy of our implemented MultinomialNB is almost the same as that of sklearn.  
However, regarding running time, our implemented MultinomialNB is slower (2-3 minutes on the whole test dataset) than that of sklearn (<1 minute).