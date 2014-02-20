#!/bin/bash
java -cp weka-3-6-10/weka.jar weka.classifiers.bayes.NaiveBayesMultinomial -t train/spam_train_1000.arff -T test/spam_test.arff
