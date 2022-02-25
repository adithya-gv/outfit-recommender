# Outfit Recommender
Group 40's CS 4641 Project: a clothing recommendation system.

Project by: Jackson Isenberg, Rishabh Jain, Rahi Kotadia, Adithya Vasudev, and Vivek Vijaykumar

## Introduction
Clothing shopping is something we do in our everyday lives, and yet it still remains a very difficult and personal decision whenever we go shop. However, with the way we shop, our history of shopping, and the similarities and differences between the clothing items themselves, this problem seems ripe for automation and machine learning to provide suggestions and alter the way we shop. Given the growth of online shopping during the pandemic, there has been a surge in technologies that have worked towards targeting the fashion industry, particularly leveraging data about customer habits and product appearances to make smart recommendations. 


### Background
Recommendation Systems have been a well-researched topic for years, and have been created to deal with vision, language, or tabular data. The fashion field has been shaped by these data types to create recommendation systems. The obvious approach would be image-centric. This entails incorporating color, shape, and other visual cues to either recommend similar clothing or complementary pieces of fashion to a given product via Neural Networks [1,2]. Another popular technique utilizes NLP (Natural Language Processing) to help identify, cluster, and recommend clothing based on the names and descriptions on various products [3]. Other sources have worked with multi-label classification for clothing or data from professional stylists to better recommend specific clothing fashion to customers [4], but these diverge from our main focus.

### Dataset Overview
The dataset we plan to use is the H&M clothing purchase history dataset (https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/overview). The dataset has both text data of over 1 million logged clothing transactions, each attributed to a given ID representing a distinct person. The dataset also has metadata for clothing and images of each clothing item.

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
For the clustering part of the project, it is likely that clothing with similar colors will be grouped. This is the most obvious way for the algorithm to perform. We can shift the focus away from color by converting images to grayscale so that we can see groupings based on clothing type, such as shirt/pants/shorts. The recommendation system should be able to take in a customer's purchase history of clothing and recommend items for them to purchase, ideally based not just on similarity but on learned attributes about what might fit well with prior purchases and what the customer might like. 

## References

1.  "Aesthetic-based Clothing Recommendation" (https://arxiv.org/pdf/1809.05822.pdf) 
2. "Image Based Fashion Product Recommendation with Deep Learning" (https://arxiv.org/pdf/1805.08694.pdf)
3. "Towards Fashion Recommendation: An AI System for Clothing Data Retrieval and Analysis" (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7256565/)
4. "Understanding Professional Fashion Stylists’ Outfit Recommendation Process: A Qualitative Study" (https://www.researchgate.net/publication 350315907_Understanding_Professional_Fashion_Stylists'_Outfit_Recommendation_Process_A_Qualitative_Study)


## Timeline
| GANTT CHART                         |                                     |            |          |          |
|-------------------------------------|-------------------------------------|:----------:|:--------:|:--------:|
|                                     |                                     |            |          |          |
|                       PROJECT TITLE | Outfit Recommender Project Timeline |            |          |          |
|                                     |                                     |            |          |          |
|                                     |                                     |            |          |          |
|              TASK TITLE             |              TASK OWNER             | START DATE | DUE DATE | DURATION |
|                                     |                                     |            |          |          |
| Project Team Composition            | All                                 |   1/17/22  |  2/1/22  |    14    |
| PROJECT PROPOSAL                    |                                     |            |          |          |
| Introduction & Background           | Vivek, Adithya                      |   2/2/22   |  2/24/22 |    22    |
| References                          | Vivek                               |   2/2/22   |  2/24/22 |    22    |
| Problem Definition                  | Rahi                                |   2/2/22   |  2/24/22 |    22    |
| Timeline                            | Rishabh/Rahi                        |   2/2/22   |  2/24/22 |    22    |
| Methods                             | Adithya                             |   2/2/22   |  2/24/22 |    22    |
| Potential Dataset                   | Vivek                               |   2/2/22   |  2/24/22 |    22    |
| Potential Results & Discussion      | Rishabh                             |   2/2/22   |  2/24/22 |    22    |
| Video Creation & Recording          | Jackson                             |   2/2/22   |  2/24/22 |    22    |
| GitHub Page                         | Adithya                             |   2/2/22   |  2/24/22 |    22    |
| MIDTERM REPORT                      |                                     |            |          |          |
| Clustering Algorithm Selection (M1) | All                                 |   2/25/22  |  2/28/22 |     3    |
| M1 Data Cleaning                    | Rahi                                |   2/25/22  |  3/4/22  |     9    |
| M1 Data Visualization               | Rishabh                             |   2/25/22  |  3/4/22  |     9    |
| M1 Feature Reduction                | Jackson                             |   2/25/22  |  3/4/22  |     9    |
| M1 Implementation & Coding          | Adithya                             |   3/5/22   |  3/14/22 |     9    |
| M1 Results Evaluation               | All                                 |   3/15/22  |  3/18/22 |     3    |
| Recommendation Alg Selection (M2)   | All                                 |   2/25/22  |  2/28/22 |     3    |
| M2 Data Cleaning                    | Vivek                               |   3/19/22  |  4/4/22  |    15    |
| M2 Data Visualization               | Rishabh                             |   3/19/22  |  4/4/22  |    15    |
| M2 Feature Reduction                | Adithya                             |   3/19/22  |  4/4/22  |    15    |
| Midterm Report                      | All                                 |   3/19/22  |  4/5/22  |    16    |
| FINAL REPORT                        |                                     |            |          |          |
| M2 Coding & Implementation          | Jackson                             |   4/5/22   |  4/19/22 |    14    |
| M2 Results Evaluation               | All                                 |   4/20/22  |  4/21/22 |     1    |
| M1-M3 Comparison                    | All                                 |   4/22/22  |  4/24/22 |     2    |
| Video Creation & Recording          | All                                 |   4/18/22  |  4/26/22 |     8    |
| Final Report                        | All                                 |   4/18/22  |  4/26/22 |     8    |
