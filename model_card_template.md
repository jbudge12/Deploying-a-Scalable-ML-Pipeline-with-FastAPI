# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model type is a logistic regresssion powered by scikitlearn. The objective is to use variables to predict whether or not an individual makes more than 50k. The 50k target, in this case, is our boolean target variable (do they make more than 50k y/n).
Model date 1/16/2025
Model version 1.0.0
Send questions to jbudge3@my.wgu.edu

## Intended Use

The primary intended use is to automate the machine learning output of the data set using FastAPI. Intended users are all those who are curious about the factors that weigh into predicting whether or not an individual makes more than 50k. 
Out of scope uses; using datasets outside the original dataset. Our predictions and outcomes are influenced by the existing variables and how the ML algorithm was trained around them.

## Training Data

The training data is the attached census data CSV. This data was split into training and testing sets and modeled around what was found via this split. 

## Evaluation Data

There were more than 30k test data. The evaluation followed the same process for preprocessing as it did the training data.

## Metrics

Overall metrics
Precision: 0.7616 Recall: 0.2094 F1: 0.3285

## Ethical Considerations

Included in the data set is country of origin, race, salary, etc. This type of data needs to be handled on a need-to-know basis. The data should be stored and used securely. 

## Caveats and Recommendations

There are many factors that go into what makes a person. One should not rely on this data set along to make a decision about whether or not to give an individual a loan, for example. Credit history, personality, background, etc can all play a role in determining if someone is a good candidate or not.
