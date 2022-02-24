# Outfit Recommender
Group 40's CS 4641 Project: a clothing recommendation system.

Project by: Jaxon Isenberg, Rishabh Jain, Rahi Kotadia, Adithya Vasudev, and Vivek Vijaykumar

## Introduction
Clothing shopping is something we do in our everyday lives, and yet it still remains a very difficult and personal decision whenever we go shop. However, with the way we shop, our history of shopping, and the similarities and differences between the clothing items themselves, this problem seems ripe for automation and machine learning to provide suggestions and alter the way we shop. 

## Background

## Dataset Overview
The dataset we plan to use is the H&M clothing purchase history dataset. The dataset has both text data of over 1 million logged clothing transactions, each attributed to a given ID representing a distinct person. The dataset also has metadata for clothing and images of each clothing item.

## Problem Statement

### Motivation
The motivation of our project was to bring the recommendation system algorithms, which have pervaded our media world, to an area of shopping that is highly personal and varied: clothes shopping. A secondary motivation was to go beyond simple 
analysis of purchasing data and use data intrinsic to the clothing data and user data to provide recommendations that target particular features of the user itself.

### Our Problems
Our problem is twofold. 
1. Measure the similarity between two arbitrary pieces of clothing and use this measurement to construct groupings of similar clothings. 
2. Construct a recommender system that will recommend clothing based on their purchase history. 

## Intended Methods
Each of the two subproblems within our overall problem has distinct approaches.

### Clothing Similarity
We plan to measure clothing similarity by applying a clustering algorithm over the features of each clothing item. We plan to apply a variety of clustering algorithms, ranging from K-Means to DBSCAN, to see which captures the shape of the data the best.

### Recommender System
To create the clothing recommender system, we plan to test a variety of popular approaches. The two front-runners for our recommender system so far are an artificial neural network, and a support vector machine. 

## Potential Results
For the clustering part of the project, it is likely that clothing with similar colors will be grouped. This is the most obvious way for the algorithm to perform. We hope to also see groupings based on clothing type, such as shirt/pants/shorts. The recommendation system should be able to take in a customer's purchase history of clothing and recommend items for them to purchase, ideally based not just on similarity but on learned attributes about what might fit well with prior pruchases and what the customer might like. 

## References

## Timeline
