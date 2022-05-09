# SANOJ DODDAPANENI
### Project 3 - The Unredactor
**Introduction** - In this project, we need to take the dataset unredactor.tsv which is built by the while class, perform necessary cleaning of the data, apply vectorization to the cleaned data, modeling the cleaned traning and validation data and then predicting the test data accordingly. We will have to return Accuracy, Precision, F1 and Recall scores in the project and print them to the output accordingly.

This project is developed using python and command line tools in Ubuntu.  
#### Sources -   
**_For Data Cleaning -_** _https://machinelearningknowledge.ai/11-techniques-of-text-preprocessing-using-nltk-in-python/_  
**_Vectorization -_** _https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html_  
**_For Modeling the data -_** _https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html_  

**Installation directions -** In this project, we use the packages below -
 - pandas
 - nltk
 - sklearn
 - sys
 - os
 - wget
 - pytest
 
We can install each of these packages using the command below -  
```
pipenv install [package_name]
```

A Pipfile is already provided in the project repository and once cloned, we can use the command below to install the packages in the Pipfile -  
```
pipenv install
```

#### Project Description -
**unredactor.py file -** This file constains the functions and execution code for the program for desired output. Each function is further explained below.

We initially, use the library os and wget where we check and remove the old unredactor.tsv file exists in the directory an  download the latest unredactor.tsv file using the view raw link from the repository. This will download the file in the main directory as required. This unredactor.tsv file has 4 columns githubname, type of data, name which is redacted and sentence where the name is redacted. These columns are not labelled initially.

Once the file is downloaded, we create dataframe for the data read and assign labels accordngly. Further, apply label encoding to names column to yeild bette results.

In the next step,we have to clean the data for have better prediction and accuracy. Here, I am removing whitespaces, stopwords, punctuations and performing stemming accordingly.

**remove_whitespace() function -** Here, we pass the sentences where the name is redacted and the function will remove the whitespaces using join() and split() functions which are inbuilt.

**remove_stopwords() function -** This function takes the returned output from remove_whitespace() function and then remove the stopwords which are listed in stopwords.words of nltk.corpus library.

**remove_punct() function -** This will take the input from the remove_stopwords() function and remove any puncuations using RegexpTokenizer library in nltk.tokenize library which we have imported accordingly.

**stemming() function -** This function takes the input from the previous function and then peform stemming which is process of natural language processing technique that lowers inflection in words to their root forms, hence aiding in the preprocessing of text, words, and documents for text normalization. This function uses PorterStemmer() library in nltk.stem library.

After performing the above cleaning process for the sentences in the dataframe, we concat the new cleaned sentences into our dataframe for ease access and further building of the model.

Now, we seperate the rows in the dataframe using .loc in pandas dataframe according to their datatype such as training, validation and testing.

We sperate and concat the results of training and validation into one dataframe for training model and then testing dataframe to predict the resulting scores accordingly.

Once the data is sperated and seggrigated accordingly, we now peform vectorization using TfidfVectorizer() function of sklearn library and then fit and transform the data to change the data into vectors to build the model and predict the results.

Now, using DecisionTreeClassifier() model, we fit the taining cleaned sentences to their names associated to in the data. Once training is completed, we use predict() function of DecisionTreeClassifier() model to predict the results on the testing data which has not included during the model.fit of training data as we should not train the testing data.

Once the predict function is used and executed, we now print the scores - Accuracy, F1, Precision and Recall accordingly using accuracy_score, precision_score, f1_score and  recall_score in the library sklearn.metrics library.

This will print us with the reslting scores to the output terminal in the python environment which numerical scores accordingly.

To run the function, we need to run the command below in main directory -  
```
pipenv run python unredactor.py
```

**Note -** _The unredactor.py program needs atleast 2GB Memory in the instance to run and could take up to 10 minutes to execute and print the results accordingly._

#### Test Cases - 
Here, we have created a new directory called **tests** and then created a file called **test_code.py** which contains different functions of test cases for each function in redactor.py. Each test case in the file is explained below accordingly.

Firstly, we import the packages sys to execute test file for all the directories of the project and provide the path accordingly and then import unredactor.py within the folder and then finally we should import package pytest to run testcases accordingly. Pyest modules works on pytest framework and can be installed using the command below -  
```
pipenv install pytest
```

**test_readinput() function -** In this function, we test if there are existing files in  folder to be read as input. the test case will be passed if there are existing files for processing.

**test_removewhile() function -** In this function, we give a string with space and assert to pass the case when it returns the same string without space.

**test_stopwords() function -** In this function, we give a string which is a stopword and assert to pass the case when it returns empty list.

**test_punc() function -** In this function, we give a string which is a punctuation and assert to pass the case when it returns empty list.

Here, Test cases can be executed using below command -  
```
pipenv run python -m pytest
```

Once the command is passed, it will show the execution of test cases.

**Note -** _The test_code.py or test cases program needs atleast 2GB Memory in the instance to run and could take up to 10 minutes to execute and validate the test case results accordingly._

#### Assumptions and Possible Bugs -  
In the unredactor.py program, my assumption is that the data is not cleaned and we need to consider the data with the label training and validation and model according and then test the trained model on testing labeled data. Changes to the data in unredactor.tsv file might result in different values for the scores.

At the end, these files are added, committed and pushed into git hub using git commands accordingly for each file.
