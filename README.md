# Emotion_based_sentiment_analysis
Emotion based sentiment analysis using multiple machine learning algorithms.

### Exaplaantion of functionality
The application is made up of multiple parts but all co inhabit a single python class.

* There are a total of 7 machine learning algorithms, they are used either individually or in conjunction with one another to sanity check results.
 * Random Forest
 * Support Vector Classification
 * Linear Regression (Multinominal Naive Bayes)
 * Logistic Regression
 * Multi Layer Perceptron Neural Network
* The application currently checks for 6 emotions
 * Anger
 * Happpy
 * Sad
 * Fear
 * Surity
 * Neutral
* The application can handle, audio data / text data and a blend of both.
* The application is using the streamlit framework to create a web spplication.

### Steps to re-produce

* Step One: Instanll virtualenv on your machine if this is not already present using command ( pip3 install virtualenv ).

* Step Two: Open a terminal and navigate into the working directory.

* Step Three: Using command ( source emotion_environment/bin/activate ) activate the virtual environment.

* Step Four: Using command ( streamlit run app.py ) start the applications front end.

* Step Five: Navigate to the specified localhost address in your terminal.

If for some reason there is a problem with the virtual enviroment, please follow the step below, these steps should be completed
in between step Two and step Three, once these steps have been completed, then resume the steps above as normal.

* Sub Step One: Using command ( virtualenv emotion_environment ) we create the virtual environment in which the application will be ran.

* Sub Step Two: Once the above step is complete we use command ( pip3 install -r requirements.txt ), this will install the requirements for the application, 
  I.e. the packages needed for the code to run properly.

The above list of steps is of course an optimal use case, if there are any errors or issues whilst running the application, 
one could install all requirements individually as if there is going to be an isues, this is primarily where it may occur.
