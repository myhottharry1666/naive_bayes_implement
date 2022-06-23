"""
Name: Hangliang Ren

This program tests the Multinomial Naive Bayes Classifier (MultinomialNB),
implemented by myself.
To do so, we build a MultinomialNB model by sklearn built-in class, and another
model by my implemented class, both through fetch_20newsgroups dataset from sklearn.
Then we will compare the accuracy of two models, calculated by predicting test dataset.

Note: My MultinomialNB is a bit slow regarding runtime, so feel free to adjust
the number of times to run in main() function (run on the whole test dataset for
one time takes around 2-3 minutes).
"""
import naive_bayes_by_myself as nbm

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np


def predict_by_sklearn_model(times_to_run):
    """
    Run test dataset on sklearn model for "times_to_run" times, record every 
    time's accuracy.
    """
    accuracies = []

    # download dataset
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)

    for i in range(times_to_run):
        # calculate tf frequency of each word
        # tf frequecy of a word = frequency of this word / total number of words in text
        count_vect = CountVectorizer()
        dataset_counts = count_vect.fit_transform(twenty_dataset.data)
        tf_transformer = TfidfTransformer(use_idf=False).fit(dataset_counts)
        dataset_tf = tf_transformer.transform(dataset_counts)
        dataset_tf = dataset_tf.todense()

        # split the dataset into train and test subset
        x_train, x_test, y_train, y_test = train_test_split(dataset_tf, twenty_dataset.target, test_size=0.3)

        # train Multinomial Naive Bayes Classifier
        clf = MultinomialNB().fit(x_train, y_train)

        # evaluate trained model by test dataset
        predicted = clf.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        accuracies.append(accuracy)
        print("sklearn model accuracy:", accuracy)
    
    return accuracies


def predict_by_my_model(times_to_run):
    """
    Run test dataset on my model for "times_to_run" times, record every 
    time's accuracy.
    """
    accuracies = []

    # download dataset
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42)

    for i in range(times_to_run):
        # split the dataset into train and test subset
        x_train, x_test, y_train, y_test = train_test_split(twenty_dataset.data, twenty_dataset.target, test_size=0.3)

        # train Multinomial Naive Bayes Classifier
        model = nbm.MultinomialNB()
        model.train(x_train, y_train)

        # evaluate trained model by test dataset
        predicted = model.predict(x_test)
        accuracy = np.mean(predicted == y_test)
        accuracies.append(accuracy)
        print("my model accuracy:", accuracy)
    
    return accuracies


def plot_accuracies(times_to_run, sklearn_acc, my_acc):
    """
    Plot accuracy from sklearn model and my model for comparison.
    """
    epochs = [i for i in range(1, times_to_run + 1)]
    plt.plot(epochs, sklearn_acc, label="sklearn model")
    plt.plot(epochs, my_acc, label="my model")
    plt.scatter(epochs, sklearn_acc, label="sklearn model, each")
    plt.scatter(epochs, my_acc, label="my model, each")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def main():
    # adjust times to run here
    times_to_run = 5

    sklearn_acc = predict_by_sklearn_model(times_to_run)
    my_acc = predict_by_my_model(times_to_run)
    plot_accuracies(times_to_run, sklearn_acc, my_acc)


if __name__ == "__main__":
    main()