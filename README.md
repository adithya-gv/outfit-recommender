# Outfit Recommender
Group 40's CS 4641 Project: a clothing recommendation system.

Project by: Jaxon Isenberg, Rishabh Jain, Rahi Kotadia, Adithya Vasudev, and Vivek Vijaykumar

## Introduction
Clothing shopping is something we do in our everyday lives, and yet it still remains a very difficult and personal decision whenever we go shop. However, with the way we shop, our history of shopping, and the similarities and differences between the clothing items themselves, this problem seems ripe for automation and machine learning to provide suggestions and alter the way we shop. 

(Alternative intro)
Clothing shopping is something we do in our everyday lives, and yet it still remains a very difficult and personal decision whenever we go shop. Given the growth of online shopping during the pandemic, there has been a surge in technologies that have worked towards targeting the fashion industry, particularly leveraging data about customer habits and product appearances to make smart recommendations. Given the complexity of the process of how customers decide what to buy, it's become essential to combine different techniques to generate the best predictions. Leveraging both textual and image data along with customer records enables practitioners to find the similarities between clothing and accordingly provide suggestions to help alter the way we shop. This problem seems ripe for automation and machine learning and is an evolving sector in the industry. 


### Background
Recommendation Systems have been a well-researched topic for years, and have been created to deal with vision, language, or tabular data. The fasion field has been shaped by these data types to create recommendation systems. The obvious approach would be image-centric. This entails incorporating color, shape, and other visual cues to either recommend similar clothing or complementary pieces of fashion to a given product via Neural Networks [1,2]. Another popular technique utilizes NLP (Natural Language Processing) to help identify, cluster, and recommend clothing based on the names and descriptions on various products [3]. Other sources have developed systems based on data collected from professional stylists to better recommend specific clothing fashion to customers [4], or have worked with multi-label classification neural networks to assign features of clothing, but these diverge from our topic of interest.

### Dataset Overview
The dataset we plan to use is the H&M clothing purchase history dataset. The dataset has both text data of over 1 million logged clothing transactions, each attributed to a given ID representing a distinct person. The dataset also has metadata for clothing and images of each clothing item.


## Problem Statement

### Motivation
The motivation of our project was to bring the recommendation system algorithms, which have pervaded our media world, to an area of shopping that is highly personal and varied: clothes shopping. A secondary motivation was to go beyond simple analysis of purchasing data and use data intrinsic to the clothing data and user data to provide recommendations that target particular features of the user itself.

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
For the clustering part of the project, it is likely that clothing with similar colors will be grouped. This is the most obvious way for the algorithm to perform. We can shift the focus awway from color by converting images to grayscale so that we can see groupings based on clothing type, such as shirt/pants/shorts. The recommendation system should be able to take in a customer's purchase history of clothing and recommend items for them to purchase, ideally based not just on similarity but on learned attributes about what might fit well with prior pruchases and what the customer might like. 

## References

1.  "Aesthetic-based Clothing Recommendation" (https://arxiv.org/pdf/1809.05822.pdf) 
2. "Image Based Fashion Product Recommendation with Deep Learning" (https://arxiv.org/pdf/1805.08694.pdf)
3. "Towards Fashion Recommendation: An AI System for Clothing Data Retrieval and Analysis" (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7256565/)
4. "Understanding Professional Fashion Stylists’ Outfit Recommendation Process: A Qualitative Study" (https://www.researchgate.net/publication 350315907_Understanding_Professional_Fashion_Stylists'_Outfit_Recommendation_Process_A_Qualitative_Study)


## Timeline
