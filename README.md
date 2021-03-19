# CSE-3521-Irony-Detection
# Daniel Attia – attia.22
# Ryan Baxter – baxter.243
# Jonathan Chi – chi.171
# Devan Mallory – Mallory.115
# Yiming Liu – liu.8672

A project for the CSE 3251 course for building an irony detection algorithm to detect irony in tweets, per https://competitions.codalab.org/competitions/17468

Dataset: https://github.com/Cyvhee/SemEval2018-Task3/tree/master/datasets

The data provided is already split into train and test data. Within the train dataset, there are three styles of the data, one plain, one containing emojis, and the last having emojis and hashtags. The train data also contains indexes for both tasks required by this competition to use for testing the implemented algorithms against for accuracy. The total size of the data set is around 4500 lines of tweets, split 3800 for training and 700 for testing. The test data, like the train data, has a plain version of the data and another with emojis. These different styles for training and testing should give us a good dataset to test our algorithms against and ensure accuracy.
