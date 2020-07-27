# Import the relevant modules.
# Primarily used for data pre-processing.
import re

import os
import math
import time
import pickle
import random
import librosa
import itertools
import unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
from typing import List, Optional
# Streamlit is the API used for everything front end,
# it is used to display information, titles and graphics to the user.
import streamlit as graphical_interface

import soundfile as sf

import plotly as py
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB


from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, hamming_loss, mean_absolute_error, mean_squared_error, mean_squared_error

# Using the utility classes, one can create more intuitive and interactive content
# for streamlit as one can use CSS and HTML natively within the window.
# Imports the grid utility class.
from utilities.layout.Grid import Grid
# Imports the cell utility class.
from utilities.layout.Cell import Cell

# Default selection screen
def no_method_selected():
    # Display the main application title to the user.
    graphical_interface.header("Multi Algorithm Sentiment Analysis For Text And Audio")
    # Using streamlits write function to force a new line.
    graphical_interface.write("\n")
    # Display information to the user.
    graphical_interface.markdown("### You can either use the side bar to select an action or navigate the training and testing data used throught the application below.")
    # Using streamlits write function to force a new line.
    graphical_interface.write("\n")
    # Display information to the user, I.e. prompt them to do something.
    graphical_interface.write("Navigate traning data")
    # Instantiate variable data_folder_directory_path and set it equal to the file path directory of the data.
    data_folder_directory_path = 'utilities/data/proccessed_data/train_data/'
    # Instantiate variable files_to_be_selected and set it data_folder_directory_path using the listdir function from os.
    files_to_be_selected = os.listdir(data_folder_directory_path)
    # Instantiate variable selected_filename and set it equal to streamlits selectbox function.
    selected_filename = graphical_interface.selectbox("Select A file", files_to_be_selected)

    # Instantiate variable text_filename and set it equal to the result of path join with folder_pathand selected_filename.
    text_filename = os.path.join(data_folder_directory_path, selected_filename)
    graphical_interface.info("You Selected {}".format(text_filename))

    # Read Data from text_filename and set selected_file_dataframe equal to the result.
    selected_file_dataframe = pd.read_csv(text_filename)

    # Selecte number of views to display.
    number = graphical_interface.number_input("Number of Rows to View", min_value=10, step=1)
    # Using streamlits dataframe function, we display the head of a dataframe which holds the rows selected by the user.
    graphical_interface.dataframe(selected_file_dataframe.head(int(number)))

    # Select Columns to show the data, I.e. label.
    all_columns = selected_file_dataframe.columns.tolist()
    # Instantiate variable selected_columns and set it equal to streamlits multi select
    # function and set all_columns to the displayed data.
    selected_columns = graphical_interface.multiselect("Select", all_columns)
    # Instantiate variable data_frame_columns_selected and set it equal to the columsn selected

    # by the user from the above multi select function.
    data_frame_columns_selected = selected_file_dataframe[selected_columns]
    # Display the result of the selected columns being selected to the user.
    graphical_interface.dataframe(data_frame_columns_selected)

# Method responsible for gathering the required data and displaying it as a bar chart to the user.
def display_bar_chart(training_data_to_display, list_of_emotions):
    # Sets array_length equal to the length of the data.
    array_length = np.arange(len(list_of_emotions))
    # Sets the width for the verticle bars within the bar plot.
    
    # Instantiate a subplots object and set it to fig and ax.
    fig, ax = plt.subplots()
    # Gets the data by columns name, I.e. emotion
    # Sets the plot to bar chart
    # and sets the legends text.
    training_data_to_display.groupby('emotion_labels').emotion_labels.count().plot.bar(ylim=0, label="Emotions")
    # Add some text for labels, title and custom x-axis tick labels, etc.
    # Sets the label for the Y axis of the bar chart.
    ax.set_ylabel('Counts')
    # Sets the title for the bar chart.
    ax.set_title('Emotions and there counts.')
    # Sets the x ticks equal to the array length, I.e. the number of emotions.
    ax.set_xticks(array_length)
    # Sets the x tick lables equal to the emotions to be represented.
    ax.set_xticklabels(list_of_emotions)
    # Adds the legend to the bar chart.
    ax.legend()
    # Displays the bar chart withijn the graphical user interface, I.e. web page.
    graphical_interface.pyplot(plt)

# Plot heatmap utility, this is used to minimise code, as within each method a single line of code can
# represent a heatmap as a confusion matrix.
def confusion_heatmap_plotting_utility(data_to_be_used, prediction_probabilities, emotion_keys):
    # Set x equal to the matrix data.
    matrix_data = confusion_matrix(data_to_be_used, prediction_probabilities)
    # Set the x value to the emotion_keys list.
    x_axis_data_labels = emotion_keys
    # Set the y value to the emotion_keys list, using emotion_keys[::-1] we flip the order of the list, 
    # to make displaying make more sense.
    y_axis_data_labels = emotion_keys
    # Set the z_text to the matrix, I.e. displays the value for each cell.
    z_axis_text = matrix_data
    # Instantiate fig and set it eqaul to the variables above and set the chart to create_annotated_heatmap
    # Along ith the color scheme.
    fig = ff.create_annotated_heatmap(matrix_data, x=x_axis_data_labels, y=y_axis_data_labels, annotation_text=z_axis_text, colorscale='Cividis')
    
    # Updates the fiogures layout with a title, ticks and axis labels, I.e. predicted labels and true labels.
    fig.update_layout(
        title='Heatmap as a Confusion Matrix',
        xaxis_nticks=36,
        xaxis_title="predicted Labels",
        yaxis_title="True Labels")

    # Ensure that the x axis is allways located along the bottom of the plot.
    fig['layout']['xaxis']['side'] = 'bottom'
    # Using streamlits functionality, Plots the figure to the screen for the user.
    graphical_interface.plotly_chart(fig)

# Method responsible for returning the information for displaying the results from running an algorithm
# I.e. scores and confusion matric plots.
def display_results_from_testing_utility(data_to_be_used, prediction_probabilities, has_length_available, averaging_to_be_used):
    # This checks to see if has_length_available is True or False,
    # The check is needed as with support vector classification
    # argmax returns to prediction which breaks len, using this
    # we check if the returned value is True, if not the support vector clasificaiton
    # is being used so we negate the argmax.
    if(has_length_available == True):
        # Sets prediction to the prediction_probabilities variable using argmax.
        prediction = np.argmax(prediction_probabilities, axis=-1)
    else:
        # Sets prediction to the plain probabilities passed in.
        prediction = prediction_probabilities

    # Returns the accuracy.
    function_accuracy_score = 'Accuracy Score: = {0:.3f}'.format(accuracy_score(data_to_be_used, prediction))
    # Returns the F score.
    function_f_one_score = 'F-One-score: = {0:.3f}'.format(f1_score(data_to_be_used, prediction, average=averaging_to_be_used))
    # Returns the precision.
    function_precision_score = 'Precision Score: = {0:.3f}'.format(precision_score(data_to_be_used, prediction, average=averaging_to_be_used))
    # Returns the recall.
    function_recall_score = 'Recall Score: = {0:.3f}'.format(recall_score(data_to_be_used, prediction, average=averaging_to_be_used))
    # Returns the mean absolute error.
    mean_absolute_error_score = 'Mean Absolute Error: = {0:.3f}'.format(mean_absolute_error(data_to_be_used, prediction))
    # Returns the mean squared error.
    mean_sqaured_error_score = 'Mean Squared Error: = {0:.3f}'.format(mean_squared_error(data_to_be_used, prediction))
    # Returns the root mean sqaured error.
    root_mean_sqaured_error_score = 'Root Mean Squared Error: = {0:.3f}'.format(np.sqrt(mean_squared_error(data_to_be_used, prediction)))

    # Returns all of the information from the variables above to be used elsewhere within the application.
    return function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score

# Method responsible for returning the bar chart plots for the training and testing data.
def display_bar_charts_for_data(testing_data_y, training_data_y, emotion_keys):
    # # For testing data
    # Instantiate data_for_testing and set it equal to testing_data_y.
    data_for_testing = testing_data_y
    # Instantiate data_for_testing_counts and set it equal to testing_data_y's value_counts.
    data_for_testing_counts = testing_data_y.value_counts()
    # Instantiate fig and set it equal to plotly express and pass the relevant variables, I.e.
    # The data to display = data_for_testing
    # x = to the emotions.
    # y = to the counts of testing_data_y value_counts.
    fig = px.bar(data_for_testing, x=emotion_keys, y=data_for_testing_counts)
    # Using fig we call update_layout to update certain elements of the plot I.e. title, xticks and the x and y tick labels.
    fig.update_layout(
        title='Emotion Counts Bar Chart (testing data)',
        xaxis_nticks=36,
        xaxis_title="Emotions",
        yaxis_title="Emotion Counts")
    # Using the graphical_interface (streamlit) we show the fig to the user.
    graphical_interface.plotly_chart(fig)

    # # For training data
    # Instantiate data_for_training and set it equal to training_data_y.
    data_for_training = training_data_y
    # Instantiate data_for_training_counts and set it equal to training_data_y's value_counts.
    data_for_training_counts = training_data_y.value_counts()
    # Instantiate fig and set it equal to plotly express and pass the relevant variables, I.e.
    # The data to display = data_for_testing
    # x = to the emotions.
    # y = to the counts of testing_data_y value_counts.
    fig = px.bar(data_for_training, x=emotion_keys, y=data_for_training_counts)
    # Using fig we call update_layout to update certain elements of the plot I.e. title, xticks and the x and y tick labels.
    fig.update_layout(
        title='Emotion Counts Bar Chart (training data)',
        xaxis_nticks=36,
        xaxis_title="Emotions",
        yaxis_title="Emotion Counts")
    # Using the graphical_interface (streamlit) we show the fig to the user.
    graphical_interface.plotly_chart(fig)

# Method responsible for taking the directory (emotion_evaluation) and generating a csv file holding the emotion labels.
def pre_process_textual_data():
    # Using streamlits markdown functionality, we display the title to the user.
    graphical_interface.markdown("### Pre Process Textual Data")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("A data directory already exists, it is advised to utilize this.")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("Once the button is pressed the system will run pre processing steps and generate a csv file.")
    # Using streamlits functionality, we instantiate a button for the user to interact with.
    run_steps_button = graphical_interface.button("Run Pre processing")
    # If the user selects the button then we run the pre processing code.
    if(run_steps_button):
        # Using regular expressions, we instantiate a varible and set it to re' compile fuinction and pass certain parameters, 
        # I.e. \n in for a new line 
        # [.+\] is match any charater within a list of readout
        # re.IGNORECASE is exactly what it sounds like, it ensure caseing is ignored.
        info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

        # Instantiate multiple arrays for use within the for loop below.
        # Each will hold a specific type of data asnd will be added to arrays for use when creating a csv file.
        times_start, times_end, individual_file_names, emotion_labels, values, value_act, value_dom = [], [], [], [], [], [], []

        # Using streamlit API's functionality to display a notice to the user that something is happening.
        with graphical_interface.spinner('Pre-processing WAV Files...'):
            # Currently using range(1, 3) / (This will pull directorys 1, 2), this is to minimise system resource use 
            # and the length of time taken to process the data, one could replace this with either a different range 
            # or an array [1, 2, 3, 4, 5] which would enable specific directory targeting/selection.
            for index in range(1, 3):
                # Instantiate a variable and set it to the emo_evaluation directory.
                # The {} denotes the are for the data from the format function at the end of the directory path, which
                # will be passed in from index from the for loop.
                emo_evaluation_dir = 'utilities/data/pre_proccessed_data/full_data_set/data_directory_{}/dialog/emo_evaluation/'.format(index)

                # Iterate through the files in each directory.
                evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
                # For loop which iterates through all files held in the evaluation_files variable as file.
                for file in evaluation_files:
                    # Open each file and readout it's contents into the variable file_content.
                    with open(emo_evaluation_dir + file) as f:
                        # Read the content from each file into the variable file_content.
                        file_content = f.read()
                    # Instantiate variable information_from_file and set it eqaul to info_line, file_content using re's findall function.
                    information_from_file = re.findall(info_line, file_content)

                    # Iterate through the information from information_from_file, I.e. using info_line and the file content preprocessed above.
                    # The first line of every iteration is counted as a header.
                    for line in information_from_file[1:]:
                        # Instantiate variables start_end_time, individual_file_name, emotion_label, val_act_dom and detect a new line using /t use strip and splits
                        # seperates the data so that each data component can be easily identifiable.
                        start_end_time, individual_file_name, emotion_label, val_act_dom = line.strip().split('\t')
                        # Instantiate variables start_time, end_time and detects seperaters using - use splits
                        # seperates the data so that each data component can be easily identifiable.
                        # Further seperating and processing of the data.
                        start_time, end_time = start_end_time[1:-1].split('-')
                        # Instantiate variables start_time, end_time and detects seperaters using , use splits
                        # seperates the data so that each data component can be easily identifiable.
                        # Further seperating and processing of the data.
                        values_act, values_act, values_dom = val_act_dom[1:-1].split(',')
                        # Sets the variables values_act, values_act, values_dom to there data counter parts and converts all to floats.
                        values_act, values_act, values_dom = float(values_act), float(values_act), float(values_dom)
                        # Sets the variables start_time, end_time to there data counter parts and converts all to floats.
                        start_time, end_time = float(start_time), float(end_time)
                        # Adds all start times to the time start array.
                        times_start.append(start_time)
                        # Adds all end times to the time end array.
                        times_end.append(end_time)
                        # Adds all of the file names to the individual_file_names array.
                        individual_file_names.append(individual_file_name)
                        # Adds all of the emotion_label to the emotion_labels array.
                        emotion_labels.append(emotion_label)
                        # Adds all of the value_act to the balues array.
                        values.append(values_act)
                        # Adds all of the values_act to the value array.
                        value_act.append(values_act)
                        # Adds all values_dom to the value_dom array.
                        value_dom.append(values_dom)

            # Instantiate a varible and set it equal to a pandad dataframe.
            # While passing the columns attribute and the column names we wish to use.
            data_frame_of_original_data = pd.DataFrame(columns=['time_start', 'time_end', 'individual_file_names', 'emotion_labels', 'values', 'value_act', 'value_dom'])

            # Sets the time_start column to the information held in the time_start variable.
            data_frame_of_original_data['time_start'] = times_start
            # Sets the times_end column to the information held in the times_end variable.
            data_frame_of_original_data['time_end'] = times_end
            # Sets the individual_file_names column to the information held in the individual_file_names variable.
            data_frame_of_original_data['individual_file_names'] = individual_file_names
            # Sets the emotion_labels column to the information held in the emotion_labels variable.
            data_frame_of_original_data['emotion_labels'] = emotion_labels
            # Sets the values column to the information held in the values variable.
            data_frame_of_original_data['values'] = values
            # Sets the value_act column to the information held in the value_act variable.
            data_frame_of_original_data['value_act'] = value_act
            # Sets the value_dom column to the information held in the value_dom variable.
            data_frame_of_original_data['value_dom'] = value_dom

            # Sets all data to its columns.
            data_frame_of_original_data.tail()
            # Saves the files as a csv in a specified directory.
            data_frame_of_original_data.to_csv('utilities/data/proccessed_data/extracted_emotion_labels.csv', index=False)

            # Using streamlits' functionality, we display information to the user.
            graphical_interface.write("Product of pre processing save as extracted_emotion_labels.csv")
        # Using streamlits functionality, we display information to the user.
        graphical_interface.success("Success")

# Method responsible for building the audio vectors for the data.
def pre_process_auditory_data():
    # Using streamlits markdown functionality, we display the title to the user.
    graphical_interface.markdown("### Pre Process Auditory Data")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("Please be aware this step will take time...")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("A data directory already exists, it is advised to utilize this.")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("Once the button is pressed the system will run pre processing steps and generate a csv file.")
    # Using streamlits functionality, we instantiate a button for the user to interact with.
    run_steps_button = graphical_interface.button("Run Pre processing")
    # If the user selects the button then we run the pre processing code.
    if(run_steps_button):

        # Instantiate extracted_labels_filepath and using read_csv, read the informaiton from the extracted_emotion_labels file.
        extracted_labels_filepath = pd.read_csv('utilities/data/proccessed_data/extracted_emotion_labels.csv')
        # Instantiate pre_processed_data_filepath and set it to the directory file path for the full data set.
        pre_processed_data_filepath = 'utilities/data/pre_proccessed_data/full_data_set/'

        # Instantiate a variable and it to a target sample rate for the sound files.
        initial_target_sample_rate = 44100
        # Create an array of sound file vectors, used when truncating individual file names for sound files.
        sound_file_vectors = {}

        # Currently using range(1, 3) / (This will pull directorys 1, 2), this is to minimise system resource use 
        # and the length of time taken to process the data, one could replace this with either a different range 
        # or an array [1, 2, 3, 4, 5] which would enable specific directory targeting/selection.
        # Along with the above, we use streamlit API's functionality to display a notice to the user that something is happening.
        with graphical_interface.spinner('Pre-processing WAV Files...'):
            for file_path_index in [1]:
                # Instantiate a variable and set it to the sound_file_path directory.
                # The {} denotes the are for the data from the format function at the end of the directory path, which
                # will be passed in from index from the for loop.
                sound_file_path = '{}data_directory_{}/dialog/wav/'.format(pre_processed_data_filepath, file_path_index)
                # Instantiate a variable and set it to the directory file path of the sounds file using os's listdir function.
                original_sound_files = os.listdir(sound_file_path)
                # For loop which iterates through the original_sound_files variable which holds
                # All of the files from the original_sound_files directory.
                for original_sound_file in tqdm(original_sound_files):
                    # Try
                    try:
                        # Instantiate a variable and set it to to outcome of librosa.load and passing the relevant information about
                        # the original sound file, the information from the iteration of said files and the initial_target_sample_rate.
                        original_sound_file_vectors, _sr = librosa.load(sound_file_path + original_sound_file, sr=initial_target_sample_rate)
                        # Using original_sound_file, file_format and instantiating variable file_format, we set the variables equal
                        # to the output of the original_sound_file using split to seperate the contents by a (.).
                        original_sound_file, file_format = original_sound_file.split('.')

                        # For loop to iterate through index, ro for the extracted_labels_filepath where extracted_labels_filepath['individual_file_names'] is present
                        # in original_sound_file and then iter rows to iterate through each row present.
                        for index, row in extracted_labels_filepath[extracted_labels_filepath['individual_file_names'].str.contains(original_sound_file)].iterrows():
                            # Instantiate variables time_start, time_end, individual_file_names, emotion_labels, values, value_act, value_dom
                            # with row from the for loop each of the variables is set to a peice of data from the original data, I.e. individual file names becomes 
                            # the file name for the original data.
                            time_start, time_end, individual_file_names, emotion_labels, values, value_act, value_dom = row['time_start'], row['time_end'], row['individual_file_names'], row['emotion_labels'], row['values'], row['value_act'], row['value_dom']
                            # Sets frame_start equal to the time start multiplied by the initial target sample rate using the math function.
                            frame_start = math.floor(time_start * initial_target_sample_rate)
                            # Sets frame_end equal to the time end multiplied by the initial target sample rate using the math function.
                            frame_end = math.floor(time_end * initial_target_sample_rate)
                            # Sets the variable truncated_sound_file_vector equal to the original_sound_file_vectors[frame start (and) frame end + 1].
                            truncated_sound_file_vector = original_sound_file_vectors[frame_start:frame_end + 1]
                            # sets the individual file names index from sound_file_vectors equal to the truncated sound file vector.
                            sound_file_vectors[individual_file_names] = truncated_sound_file_vector

                    # If an error occurs, catch it.
                    except:
                        # Print an error occured along with the file name for which it occurd
                        print('An exception occured for {}'.format(original_sound_file))
        # Using streamlits functionality, we display that the latest action is complete to the user.
        graphical_interface.success('Complete!')

        # Read the contents using pandas read_csv into a variable for later use, I.e. the emotion labels.
        # labels_df = pd.read_csv(extracted_labels_filepath)
        current_audio_vectors_file = sound_file_vectors

        randomly_generated_file_name = list(current_audio_vectors_file.keys())[random.choice(range(len(current_audio_vectors_file.keys())))]
        audio_vectors_array_y = current_audio_vectors_file[randomly_generated_file_name]

        # Instantiate variables, harmonic_vector_y and percussive_vector_y and set them equal to the result of librosa's effects hpss using audio_vectors_array_y.
        harmonic_vector_y, percussive_vector_y = librosa.effects.hpss(audio_vectors_array_y)
        np.mean(harmonic_vector_y)
        # Instantiate a variable to hold the column names for which the resulting csv will implement.
        file_columns = ['individual_file_names', 'emotion_labels', 'mean_signature', 'standard_signature', 'root_sqaure_mean', 'root_sqaure_standard', 'silence', 'harmonic_resonance', 'auto_correct_max', 'auto_correct_standard']
        # Instantiate variable data_frame_of_features and set it equal to a pandas dataframe and pass the above file_columns in as the answer to the columns arguement.
        data_frame_of_features = pd.DataFrame(columns=file_columns)

        # emotions_to_be_referenced holds the emotions which will be matched with the emotion_labels within the for loop.
        emotions_to_be_referenced = {'ang': 0,
                                    'hap': 1,
                                    'exc': 2,
                                    'sad': 3,
                                    'fru': 4,
                                    'fea': 5,
                                    'sur': 6,
                                    'neu': 7,
                                    'xxx': 8,
                                    'oth': 8}

        # Read the contents using pandas read_csv into a variable for later use, I.e. the emotion labels.
        labels_df = extracted_labels_filepath
        graphical_interface.write("Reading dataframe - ", labels_df)
        # Currently using range(1, 3) / (This will pull directorys 1, 2), this is to minimise system resource use 
        # and the length of time taken to process the data, one could replace this with either a different range 
        # or an array [1, 2, 3, 4, 5] which would enable specific directory targeting/selection.
        # Along with the above, we use streamlit API's functionality to display a notice to the user that something is happening.
        with graphical_interface.spinner('Pre-processing Auditory Data...'):
            for index in [1]:
                # Instantate variable current_audio_vectors_file and set it equal to the audio vector files contents, passing there original file directory path and the number of the file passed using index,
                # using pickles load function.
                current_audio_vectors_file = sound_file_vectors
                # For loop which using index and row, iterates the labels dataframe pulling the individual_file_names column and chekcing if the index of individual_file_names contains the string Ses
                # passing the index from the for loop as the format and using iterrows to iterate through the index until the end of file occurs.
                for index, row in tqdm(labels_df[labels_df['individual_file_names'].str.contains('Ses0{}'.format(index))].iterrows()):
                    # Try.
                    try:
                        # Instantiate variable original_sound_file_name and set it equal to the for loops row and index of individual_file_names
                        original_sound_file_name = row['individual_file_names']
                        # Instantiate variable emotion_labels and set it equal to the for loops row and index of individual_file_names using the List of emotions_to_be_referenced above
                        # This checks the amotion and asigns a number based on the emotion from the file, I.e. anger = 0.
                        emotion_labels = emotions_to_be_referenced[row['emotion_labels']]
                        # Instantiates variable audio_vectors_array_y and set it equal to the current_audio_vectors_file index of original_sound_file_name.
                        audio_vectors_array_y = current_audio_vectors_file[original_sound_file_name]

                        # Instantiate array list_of_available_features and set it equal to indexes, original_sound_file_name, emotion_labels.
                        list_of_available_features = [original_sound_file_name, emotion_labels]
                        # Instantiate variable mean_signature and set it equal to the result of abs audio_vectors_array_y using numpy's mean function.
                        mean_signature = np.mean(abs(audio_vectors_array_y))
                        # Appends the mean signature to the list_of_available_features array.
                        list_of_available_features.append(mean_signature)
                        # Append the standard signature to the list_of_available_features array.
                        list_of_available_features.append(np.std(audio_vectors_array_y))
                        # Instantiate variable root_sqaure_mean_error and set it equal to the result of librosas features root mean square function and add at index 0.
                        root_sqaure_mean_error = librosa.feature.rms(audio_vectors_array_y + 0.0001)[0]
                        # Appends the root sqaure mean error using numpys mean function and passing root_sqaure_mean_error variable from 
                        # above to the list_of_available_features array.
                        list_of_available_features.append(np.mean(root_sqaure_mean_error))
                        # Appends the root sqaure mean error standard using numpys mean function and passing root_sqaure_mean_error variable from 
                        # above to the list_of_available_features array.
                        list_of_available_features.append(np.std(root_sqaure_mean_error))

                        # Instantie variable and set it to zero.
                        silence_harmonic = 0
                        # For loop which iterates through root_sqaure_mean_error.
                        for silence_index in root_sqaure_mean_error:
                            # If silence_index is less than or equal to 0.4 multiplied by numpys mean function and passing the root_sqaure_mean_error.
                            if silence_index <= 0.4 * np.mean(root_sqaure_mean_error):
                                # Increment silnece_harmonic by eqaul or 1.
                                silence_harmonic += 1
                        # Set silence_harmonic to root_sqaure_mean_error, divided by the length of root_sqaure_mean_error.
                        silence_harmonic /= float(len(root_sqaure_mean_error))
                        # Append silence_harmonic to the list_of_available_features array.
                        list_of_available_features.append(silence_harmonic)
                        # Set harmonic_vector_y eqaul to the result of librosa's effects hpss using audio_vectors_array_y at index zero.
                        harmonic_vector_y = librosa.effects.hpss(audio_vectors_array_y)[0]
                        # Append the harmonic_vector_y to list_of_available_features array using the numpy's mean function and multiply it by 1000, I.e. scale the harmonic.
                        list_of_available_features.append(np.mean(harmonic_vector_y) * 1000)

                        # Very loosly based on the pitch detection algorithm mentioned in 
                        # Kotnik, B et, al (2006, pp. 149 - 152)
                        # Please see references in the accompanying report.
                        # Instantiate variable calculated_mean_signature and set it equal to the mean_signature multipled by 45 percent.
                        calculated_mean_signature = 0.45 * mean_signature
                        # instantiate the clipped_content_array array.
                        clipped_content_array = []
                        # For loop which iterates through the audio_vectors_array_y.
                        for audio_vectors_to_loop in audio_vectors_array_y:
                            # If the audio_vectors_to_loop is greater than or euqual to the calculated_mean_signature.
                            if audio_vectors_to_loop >= calculated_mean_signature:
                                # Append the result of audio_vectors_to_loop minus the calculated_mean_signature.
                                clipped_content_array.append(audio_vectors_to_loop - calculated_mean_signature)
                            # If the audio_vectors_to_loop is less than or euqual to minus calculated_mean_signature.
                            elif audio_vectors_to_loop <= -calculated_mean_signature:
                                # Append the result of audio_vectors_to_loop plus the calculated_mean_signature.
                                clipped_content_array.append(audio_vectors_to_loop + calculated_mean_signature)
                            # If the audio_vectors_to_loop is less than the calculated_mean_signature.
                            elif np.abs(audio_vectors_to_loop) < calculated_mean_signature:
                                # Append zero to the clipped_content_array array.
                                clipped_content_array.append(0)
                        # Instantiate variable auto_corrections and set it equal to the result of librosa'a autocorrelate using a numpy's array.
                        auto_corrections = librosa.core.autocorrelate(np.array(clipped_content_array))
                        # Append the auto_corrections using numpy's max function and divided by the length of auto_corrections to list_of_available_features and multiply it by 1000, I.e. scale the auto correct max.
                        list_of_available_features.append(1000 * np.max(auto_corrections)/len(auto_corrections))
                        # auto_corr_std
                        # Append the result of auto_corrections using numpy's standard function, I.e. auto correct standard.
                        list_of_available_features.append(np.std(auto_corrections))

                        # Set data_frame_of_features equal to itself and passing a pandas dataframe of list_of_available_features, 
                        # we pass the index as file_columns but transposed
                        # and we state that index should be ignored.
                        data_frame_of_features = data_frame_of_features.append(pd.DataFrame(list_of_available_features, index=file_columns).transpose(), ignore_index=True)
                    # If an error occurs, catch it.
                    except:
                        # Print some exception occured.
                        print('Some exception occured')

                # Using streamlits' functionality, we display information to the user.
                graphical_interface.write("Product of pre processing save as audio_features.csv")
                # Save the information saved within data_frame_of_features to a csv file using the to_csv function,
                data_frame_of_features.to_csv('utilities/data/proccessed_data/audio/features/audio_features.csv', index=False)
        # Using streamlits functionality, we display to the user that the file has been saved.        
        graphical_interface.write("File saved audio_features.csv, created and saved.")
        # Using streamlits functionality, we display that the latest action is complete to the user.
        graphical_interface.success('Complete!')

# Method responsible for preparing the data Aia, taking the partially processed data and creating the training and testing data.
def process_and_prepare_data_method():
    # Using streamlits markdown functionality, we display the title to the user.
    graphical_interface.markdown("### Prepare Textual and Auditory Data")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("Please be aware this step will take time...")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("A data directory already exists, it is advised to utilize this.")
    # Using streamlits functionality, we display information to the user.
    graphical_interface.write("Once the button is pressed the system will run pre processing steps and generate a csv file.")
    # Using streamlits functionality, we instantiate a button for the user to interact with.
    run_steps_button = graphical_interface.button("Run Pre processing")

    # If the user selects the button then we run the pre processing code.
    if(run_steps_button):
        # Instantiate data_frame_of_audio_features and set it equal to the audio features files directory file path.
        data_frame_of_audio_features = pd.read_csv('utilities/data/proccessed_data/audio/features/audio_features.csv')
        # Create a dataframe of audio features and detect/declare the emotiona labels and using isin, we detect there values, I.e. 
        # 'ang': 0
        # 'hap': 1
        # 'exc': 2
        # 'sad': 3
        # 'fru': 4
        # 'fea': 5
        # 'sur': 6
        # 'neu': 7
        data_frame_of_audio_features = data_frame_of_audio_features[data_frame_of_audio_features['emotion_labels'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
        # Using streamlits functionality, we display information to the user.
        graphical_interface.write("Data Frame Of Features.")
        # Using streamlits functionality, we show the data to the user.
        graphical_interface.write(data_frame_of_audio_features)

        # Using the map function we map each emotion to it's corresponding emotion label.
        data_frame_of_audio_features['emotion_labels'] = data_frame_of_audio_features['emotion_labels'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
        
        # Using to_csv we create a file named sampled_data_frame and write the data from data_frame_of_audio_features.
        data_frame_of_audio_features.to_csv('utilities/data/proccessed_data/audio/samples/sampled_data_frame.csv')

        # Creates oversampling for the emotion (fear).
        data_frame_for_fear = data_frame_of_audio_features[data_frame_of_audio_features['emotion_labels']==3]
        # For loop to iterate through 30 samples of fear, I.e. data_frame_of_audio_features['emotion_labels']==3] corresponds to fear.
        for i in range(30):
            # set data_frame_of_audio_features equal to itself but appending data_frame_for_fear.
            data_frame_of_audio_features = data_frame_of_audio_features.append(data_frame_for_fear)

        # Creates oversampling for the emotion (surprise).
        data_frame_for_suprise = data_frame_of_audio_features[data_frame_of_audio_features['emotion_labels']==4]
        # For loop to iterate through 10 samples of fear, I.e. data_frame_of_audio_features['emotion_labels']==4] corresponds to surprise.
        for i in range(10):
            # set data_frame_of_audio_features equal to itself but appending data_frame_for_suprise.
            data_frame_of_audio_features = data_frame_of_audio_features.append(data_frame_for_suprise)
            
        # Using to_csv we create a file named modified_sampled_data_frame and write the data from data_frame_of_audio_features.
        data_frame_of_audio_features.to_csv('utilities/data/proccessed_data/audio/modified_sampled_data_frame.csv')

        # Instantiate variable min_max_auto_scaler and set it equal to sklearns MinMaxScaler sunction.
        min_max_auto_scaler = MinMaxScaler()

        # Using the min_max_auto_scaler we set the first two columns of data_frame_of_audio_features equal to the 
        # min_max_auto_scaler.fit_transform, then we pass the first two columns of data_frame_of_audio_features[data_frame_of_audio_features.columns[2:]]
        data_frame_of_audio_features[data_frame_of_audio_features.columns[2:]] = min_max_auto_scaler.fit_transform(data_frame_of_audio_features[data_frame_of_audio_features.columns[2:]])
        # Using streamlits functionality, we display information to the user.
        graphical_interface.write("Data Frame Of Features After Min Max Scaling.")
        # Using streamlits functionality, we show the data to the user.
        graphical_interface.write(data_frame_of_audio_features.head())

        # Instantiate variables, training_data_x, testing_data_x and set them equal to th product of train_test_splitusing the data_frame_of_audio_features
        # variable and set it's test size to 20% of all of the data available.
        training_data_x, testing_data_x = train_test_split(data_frame_of_audio_features, test_size=0.20)
        # Using streamlits functionality, we display information to the user about backend functionality.
        graphical_interface.write("Generating Auditory Testing And Training Data.")
        # Directory for use as training and testing data
        # Using the to_csv function we write the information from text_train to a csv file.
        training_data_x.to_csv('utilities/data/proccessed_data/audio/audio_conversion/audio_train.csv', index=False)
        # Using the to_csv function we write the information from text_test to a csv file.
        testing_data_x.to_csv('utilities/data/proccessed_data/audio/audio_conversion/audio_test.csv', index=False)

        # Directory for use as displaying to the user, I.e. data exploration.
        # Using the to_csv function we write the information from text_train to a csv file.
        training_data_x.to_csv('utilities/data/proccessed_data/train_data/audio_train.csv', index=False)
        # Using the to_csv function we write the information from text_test to a csv file.
        testing_data_x.to_csv('utilities/data/proccessed_data/train_data/audio_test.csv', index=False)
        # Using streamlits functionality, we display information to the user that it was successful.
        graphical_interface.write("Success.")

        # Using regular expressions, we instantiate a varible and set it to re' compile fuinction and pass certain parameters, 
        # (\w+) is match any charater
        # re.IGNORECASE is exactly what it sounds like, it ensure caseing is ignored.
        complete_regular_expression = re.compile(r'^(\w+)', re.IGNORECASE)

        # File to transcription array instantiation.
        file_too_transcription_conversion = {}

        # Currently using range(1, 3) / (This will pull directorys 1, 2), this is to minimise system resource use 
        # and the length of time taken to process the data, one could replace this with either a different range 
        # or an array [1, 2, 3, 4, 5] which would enable specific directory targeting/selection.
        for index in range(1, 3):
            # Instantiate the variable transcriptions_file_path and set it equal the directory file path for the transcriptions
            # data, which is used to generate the text files for train ands  test.
            transcriptions_file_path = 'utilities/data/pre_proccessed_data/full_data_set/data_directory_{}/dialog/transcriptions/'.format(index)
            # Iterate through the files in each directory.
            transcriptions_files = os.listdir(transcriptions_file_path)
            # For loop which iterates through all files held in the transcriptions_files variable as file.
            for files in transcriptions_files:
                # Open each file and readout it's contents into the variable readout_text_lines.
                with open('{}{}'.format(transcriptions_file_path, files), 'r') as files:
                    # Read the content from each file into the variable readout_text_lines.
                    readout_text_lines = files.readlines()

                # For loop which iterates through the read out text lines from above lines.
                for lines in readout_text_lines:
                    # Instantiate audio_code and set it equal to the complete complete_regular_expression.math, this match comes from the data itsels and is a type label,
                    # then we cvall group to group data together.
                    audio_code = complete_regular_expression.match(lines).group()
                    # Instantiate transcription_data and set it equal to the lines which comes from the for loop,
                    # We then string -1 : from the data.
                    transcription_data = lines.split(':')[-1].strip()

                    # We set the file_too_transcription_conversion and pass in the audio_code variable as an index and set it equal to the transcription_data.
                    file_too_transcription_conversion[audio_code] = transcription_data

        # Below we set the files for text test and train.

        # Create a pandas data fram eand set text_train equal to this data frame.
        text_train = pd.DataFrame()
        # We set text_train index (individual file names) equal to training_data_x (individual file names).
        text_train['individual_file_names'] = training_data_x['individual_file_names']
        # We set text_train index (emotion labels) equal to training_data_x (emotion labels).
        text_train['emotion_labels'] = training_data_x['emotion_labels']
        # We set text_train index (transcription) equal to the result of the normalized audio_too_text_conversion method and pass code as it's index, 
        # We then iterate through the code variables data present in training_data_x['individual_file_names']].
        text_train['transcription'] = [string_normalization_method(file_too_transcription_conversion[code]) for code in training_data_x['individual_file_names']]

        # Create a pandas data fram eand set text_test equal to this data frame.
        text_test = pd.DataFrame()
        # We set text_test index (individual file names) equal to testing_data_x (individual file names).
        text_test['individual_file_names'] = testing_data_x['individual_file_names']
        # We set text_test index (emotion labels) equal to testing_data_x (emotion labels).
        text_test['emotion_labels'] = testing_data_x['emotion_labels']
        # We set text_test index (transcription) equal to the result of the normalized audio_too_text_conversion method and pass code as it's index, 
        # We then iterate through the code variables data present in testing_data_x['individual_file_names']].
        text_test['transcription'] = [string_normalization_method(file_too_transcription_conversion[code]) for code in testing_data_x['individual_file_names']]

        # Using streamlits functionality, we display information to the user about backend functionality.
        graphical_interface.write("Generating Textual Testing And Training Data.")
        # Directory for use as training and testing data
        # Using the to_csv function we write the information from text_train to a csv file.
        text_train.to_csv('utilities/data/proccessed_data/text/text_conversion/text_train.csv', index=False)
        # Using the to_csv function we write the information from text_test to a csv file.
        text_test.to_csv('utilities/data/proccessed_data/text/text_conversion/text_test.csv', index=False)

        # Directory for use as displaying to the user, I.e. data exploration.
        # Using the to_csv function we write the information from text_train to a csv file.
        text_train.to_csv('utilities/data/proccessed_data/train_data/text_train.csv', index=False)
        # Using the to_csv function we write the information from text_test to a csv file.
        text_test.to_csv('utilities/data/proccessed_data/train_data/text_test.csv', index=False)

        # Using streamlits functionality, we display information to the user that it was successful.
        graphical_interface.write("Success.")

# Function responsible for defining a uni code to ascii conversion.
def uni_code_to_ascii_conversion(string_being_used):
    # Return a string, I.e. string_being_used which has been normalized and categorized.
    return ''.join(
        c for c in unicodedata.normalize('NFD', string_being_used)
        if unicodedata.category(c) != 'Mn'
    )

# Function responsible for trimming, removing all and any non letter characters, I.e. symbols such as ! ?.
def string_normalization_method(string_being_used):
    # We call the uni code to ascii converter and pass it a string, we call lower to decapitilize all and call strip to remove unwanted charaters.
    string_being_used = uni_code_to_ascii_conversion(string_being_used.lower().strip())
    # Using the same string we use regular expressions to further preprocess this string.
    string_being_used = re.sub(r"([.!?])", r" \1", string_being_used)
    # Using the same string we use regular expressions to further preprocess this string.
    string_being_used = re.sub(r"[^a-zA-Z.!?]+", r" ", string_being_used)
    # Return the string once all pre processing is complete.
    return string_being_used

# Training algorithms using text and audio data.
def data_and_model_selection_method():
    # Using streamlits functionality, Displa the title to the user.
    graphical_interface.title("Sentiment analyses using auditory and textual data")
    # Using streamlits functionality, Machine learning algorithm selection section.
    choose_data_type_for_algorithms = graphical_interface.selectbox("Choose an",
                                                    ["NONE", "Text",
                                                        "Audio"])

    # If none, then nothing has been selected.
    if(choose_data_type_for_algorithms == "NONE"):
        print("NONE")

    # Else if text, text has been selected.
    elif(choose_data_type_for_algorithms == "Text"):
        # Load the text files for text training into the variable text_training_testing_data.
        text_training_testing_data = pd.read_csv('utilities/data/proccessed_data/text/text_conversion/text_train.csv')
        # Append the testing data into text_training_testing_data.
        text_training_testing_data = text_training_testing_data.append(pd.read_csv('utilities/data/proccessed_data/text/text_conversion/text_test.csv'))
        # Using streamlits functionality, Small sampling of the data.
        graphical_interface.write("Displaying a small sampling of the data.")
        # Using streamlits functionality, Display the head of the variable text_training_testing_data.
        graphical_interface.write(text_training_testing_data.head())
        # Instantiate array and set it equal to emotion_labels and transcription (columns from the data).
        col = ['emotion_labels', 'transcription']
        # Set variable text_training_testing_data equal to text_training_testing_data along 
        # with the data present within the col.
        text_training_testing_data = text_training_testing_data[col]
        # Set the columns within text_training_testing_data equal to emotion_labels and transcription.
        text_training_testing_data.columns = ['emotion_labels', 'transcription']
        # Using sklearns functionality with feature_extraction we instantiate variable t_vectorizer
        # and pass in the relevant arguements, I.e. 
        # sublinear_tf Apply sub linear tensorflow scalling, I.e. replace tf with 1 + log(tf)..
        # min_df = This will ignore terms within a document frequency strictly lower the value specified.
        # norm = The output row will have a unit nrom of, ‘l2’: Sum of squares of vector elements is 1.
        # encoding = The encoding to be used, I.e. UTF, latin etc.
        # ngram_range = A tuple of lower and upper boundaries for n values.
        # stop_words = the string value supported by TfidfVectorizer.
        t_vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        # Instantiate variable features and set it equal to the resul of TfidfVectorizer's fit_transform functionality, 
        # while passing variable column transcriptions from text_training_testing_data and finally to call to array.
        features = t_vectorizer.fit_transform(text_training_testing_data.transcription).toarray()
        # Instantiate variable labels and set it equal to the emotion_labels column from text_training_testing_data.
        labels = text_training_testing_data.emotion_labels

        # Instantiate variables, training_data_x, testing_data_x, training_data_y, testing_data_y,
        # and set them equal to the results of sklearns trainb_test_split and passing the features and labels to be used along
        # with the specified test size,
        # I.e. will split the original data into test and train data, the test data will be 20% of all data.
        training_data_x, testing_data_x, training_data_y, testing_data_y = train_test_split(features, labels, test_size=0.20)

        # Using my grid API for generating a grid.
        with Grid("1 1") as grid:
            # Using the grid class we creatge a new grid element and pass it a cell type
            # the grid portion of this functionality takes a
            #
            # class this string is the identifier for the cell,
            # An int in position one which is the cells position width,
            # An int in position two which is the cells width,
            # An int in position three which is the position height,
            # An int in position four which is the cells height,

            # After this we pass a cell type, these can be text, markdown and dataframe.
            # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
            # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
            grid.cell("a", 2, 3, 3, 2).cell_text(
                "Training X Shape " + str(tuple(training_data_x.shape)))
            grid.cell("b", 1, 2, 3, 2).cell_text(
                "Testing X Shape " + str(tuple(testing_data_x.shape)))
        # Dictionary of the emotion present within the training / testing data.
        emotion_dictionary = {'ang': 0,
                              'hap': 1,
                              'sad': 2,
                              'fea': 3,
                              'sur': 4,
                              'neu': 5}
        # List of emotions which will be used to make / display a prediction on 
        # which emotion is present.
        emotion_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])
    
    # Else if choose_data_type_for_algorithms equals audio.
    elif(choose_data_type_for_algorithms == "Audio"):
        # Instantiate variable training_data_x and set it equal to the file path directory for the audio trining data.
        training_data_x = pd.read_csv('utilities/data/proccessed_data/audio/audio_conversion/audio_train.csv')
        # Instantiate variable testing_data_x and set it equal to the file path directory for the audio testing data.
        testing_data_x = pd.read_csv('utilities/data/proccessed_data/audio/audio_conversion/audio_test.csv')
        # Using streamlits functionality, Small sampling of the data.
        graphical_interface.write("Displaying a small sampling of the data.")
        # Using streamlits functionality, Display the head of variable training_data_x
        graphical_interface.write(training_data_x.head())
        # My preliminary idea of an API for generating a grid
        with Grid("1 1") as grid:
            # Using the grid class we creatge a new grid element and pass it a cell type
            # the grid portion of this functionality takes a
            #
            # class this string is the identifier for the cell,
            # An int in position one which is the cells position width,
            # An int in position two which is the cells width,
            # An int in position three which is the position height,
            # An int in position four which is the cells height,

            # After this we pass a cell type, these can be text, markdown and dataframe.
            # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
            # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
            grid.cell("a", 2, 3, 3, 2).cell_text(
                "Training X Shape " + str(tuple(training_data_x.shape)))
            grid.cell("b", 1, 2, 3, 2).cell_text(
                "Testing X Shape " + str(tuple(testing_data_x.shape)))
        # Instantiate variable training_data_y and set it equal to training_data_x column emotion_labels.
        training_data_y = training_data_x['emotion_labels']
        # Instantiate variable testing_data_y and set it equal to testing_data_x column emotion_labels.
        testing_data_y = testing_data_x['emotion_labels']

        # List of emotions which will be used to make / display a prediction on 
        # which emotion is present.
        emotion_keys = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])

        # Method which is responsible for removing any and all unwanted features from the data
        # I.e. anything which is string based from the audio data.
        def remove_unwanted_data_features():
            # Removes label from the training data.
            del training_data_x['emotion_labels']
            # Removes label from the testing data.
            del testing_data_x['emotion_labels']
            # Removes individual_file_names from the training data.
            del training_data_x['individual_file_names']
            # Removes individual_file_names from the testing data.
            del testing_data_x['individual_file_names']

        # Dictionary of the emotion present within the training / testing data.
        emotion_dictionary = {'ang': 0,
                              'hap': 1,
                              'sad': 2,
                              'fea': 3,
                              'sur': 4,
                              'neu': 5}

    # If none is selected.
    if(choose_data_type_for_algorithms == "NONE"):
        # Using streamlits functionality, Display information to the user.
        graphical_interface.write("Please select a data type for more options.")

    # Else if choose_data_type_for_algorithms  doe not equal none.
    elif(choose_data_type_for_algorithms != "NONE"):
        # Using streamlits functionality, Machine learning algorithm selection section.
        choose_algorithmic_approach = graphical_interface.selectbox("Choose a traing method",
                                                                ["NONE", "Random Forest",
                                                                 "Support Vector Classification",
                                                                 "Linear Regression (Multinominal Naive Bayes)",
                                                                 "Logistic Regression",
                                                                 "Multi Layer Perceptron Neural Network",
                                                                 "Run All Models (Beware!!, will take time)"])

        # Used to detect which algorithmic approache has been chose.
        # If nothing has been selected then we check for the which datatype has been selected
        if(choose_algorithmic_approach == 'NONE'):
            # If the user selects Text.
            if(choose_data_type_for_algorithms == "Text"):
                # Use the method display_bar_chart and send text_training_testing_data as the data source.
                display_bar_chart(text_training_testing_data, emotion_keys)
            # If the user selects Audio.
            elif(choose_data_type_for_algorithms == "Audio"):
                # Use the method display_bar_chart and send training_data_x as the data source.
                display_bar_chart(training_data_x, emotion_keys)
            # Using streamlits functionality, Dislay text to the user prompting them to select an algorithmic method.
            graphical_interface.write('Please select an algorithmic method')
            # Using streamlits functionality, Dislay text to the user with an explanantion.
            graphical_interface.write(
                'The learning process will occur immediately after a selection is made.')

        # If the user selects Random Forest.
        elif(choose_algorithmic_approach == "Random Forest"):
            print("Random Forest")

            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()
            # We call the method responisble for the random forest algorithm and pass the relevant data / information.
            random_forest_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)

        # If the user selects "Support Vector Classification".
        elif(choose_algorithmic_approach == "Support Vector Classification"):
            print("Support Vector Classification")
            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()
            # We call the method responisble for the support vector classification algorithm and pass the relevant data / information.
            support_vector_classification_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)        

        # If the user selects Multinominal Naive Bayes).
        elif(choose_algorithmic_approach == "Linear Regression (Multinominal Naive Bayes)"):
            print("Multinominal Naive Bayes")
            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()
            # We call the method responisble for the multinominal naive bayes algorithm and pass the relevant data / information.
            multinominal_naive_bayes_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)        

        # If the user selects "Logistic Regression".
        elif(choose_algorithmic_approach == "Logistic Regression"):
            print("Logistic Regression")
            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()
                # We call the method responisble for the logistic regression algorithm and pass the relevant data / information.
            logistic_regression_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)

        # If the user selects "Multi Layer Perceptron".
        elif(choose_algorithmic_approach == "Multi Layer Perceptron Neural Network"):
            print("Multi Layer Perceptron Neural Network")
            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()
            # We call the method responisble for the multi perceptron algorithm and pass the relevant data / information.
            multi_perceptron_neural_network_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)
        
        # If the user selects "Multi Layer Perceptron".
        elif(choose_algorithmic_approach == "Run All Models (Beware!!, will take time)"):
            # Used to detect which data type has been chose.
            # If Text has been chosen then we laod the data as is.
            if(choose_data_type_for_algorithms == "Text"):
                print("Text")
            # If audio has been chosen, we modifgy the data
            elif(choose_data_type_for_algorithms == "Audio"):
                # We remove the unwanted features from the data.
                remove_unwanted_data_features()

            # Using streamlits functionality, Display a which algorithmic approach was selected.
            graphical_interface.write("Random Forest Classification")
            # We call the method responisble for the random forest algorithm and pass the relevant data / information.
            forest = random_forest_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)
            # Using streamlits functionality, Display a which algorithmic approach was selected.
            graphical_interface.write("Support Vector Classification")
            # We call the method responisble for the support vector classification algorithm and pass the relevant data / information.
            vector = support_vector_classification_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms) 
            # Using streamlits functionality, Display a which algorithmic approach was selected.
            graphical_interface.write("Linear Regression Classification")
            # We call the method responisble for the multinominal naive bayes algorithm and pass the relevant data / information.
            linear = multinominal_naive_bayes_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)
            # Using streamlits functionality, Display a which algorithmic approach was selected.
            graphical_interface.write("Logistic Regression Classification")
            # We call the method responisble for the logistic regression algorithm and pass the relevant data / information.
            logistic = logistic_regression_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)
            # Using streamlits functionality, Display a which algorithmic approach was selected.
            graphical_interface.write("Multi Layer Perceptron Classification")
            # We call the method responisble for the multi layer perceptron algorithm and pass the relevan data / information.
            perceptron = multi_perceptron_neural_network_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms)

# Method responsible for training a random forest classifier.
def random_forest_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms):
    # Instantiate variable random_forest_classifier, pass the relevant information, setting it to the variable random_forest_classifier.
    # the n estimators = the actual number of trees to be instantiated for the forest.
    # the min samples split = the actual number of data points which will be added to a node before it is split.
    random_forest_classifier = RandomForestClassifier(n_estimators=1200, min_samples_split=25)

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()
    # Using variable random_forest_classifier we call sklearns fit functionality to fit the model and 
    # pass the relevant training data training_data_x, training_data_y.
    random_forest_classifier.fit(training_data_x, training_data_y)

    # End the timer for the fitting of the function
    fitting_time_end = time.time()
    # Calculate the tootal time taken by deducting fitting_time_start from fitting_time_end.
    total_fitting_time = fitting_time_end-fitting_time_start
    # Start a timer for the prediction of the function.
    prediction_time_start = time.time()

    # Instantiate the variable prediction_probabilities and using the predic function from sklearns RandomForestClassifier 
    # while passing the testing data set it equal to the outcome.
    # Making the actual prediction.
    prediction_probabilities = random_forest_classifier.predict_proba(testing_data_x)

    # End the timer for the predicting of the function
    prediction_time_end = time.time()
    # Calculate the tootal time taken by deducting prediction_time_start from prediction_time_end.
    total_prediction_time = prediction_time_end-prediction_time_start
    # Using streamlits functionality, Display information to the user.
    graphical_interface.write('Table of prediction probabilities.')
    # Using streamlits functionality, Displays a table made up from the prediction proprobabilities to the user.
    graphical_interface.write(prediction_probabilities)

    # Set to True as argmax is available for random forest 
    # --- See the notes in display_results_from_testing_utility ---.
    has_length_available = True
    # Specifies the averaging to be used by the metrics, I.e. f1 score, precision.
    averaging_to_be_used = 'macro'

    # Instanite the method display_results_from_testing_utility and set the passed variables equal
    # variables, thus we can print the informaiton to the screen.
    function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score = display_results_from_testing_utility(testing_data_y, prediction_probabilities, has_length_available, averaging_to_be_used)
    # Using streamlits functionality, Display that the training is copmplete to the user.
    graphical_interface.markdown("### Training Complete")

    # Creates a string which will be used in conjunction with total_fitting_time to display total time taken.
    fitting_string = 'Fitting took {0:.3f} seconds'
    # Creates a string which will be used in conjunction with prediction_string to display total time taken.
    prediction_string = 'Prediction took {0:.3f} seconds'
    
    # Checks if the fitting time is less than 1.
    if(total_fitting_time < 1):
        # If so then we replace the word seconds with milliseconds
        fitting_string = fitting_string.replace('seconds',' milliseconds')
    # Checks if the prediction time is less than 1.
    if(total_prediction_time < 1):
        # If so then we replace the word seconds with milliseconds
        prediction_string = prediction_string.replace('seconds',' milliseconds')

    # Using my grid API for generating a grid.
    with Grid("1 1") as grid:
        # Using the grid class we creatge a new grid element and pass it a cell type
        # the grid portion of this functionality takes a
        
        # class this string is the identifier for the cell,
        # An int in position one which is the cells position width,
        # An int in position two which is the cells width,
        # An int in position three which is the position height,
        # An int in position four which is the cells height,

        # After this we pass a cell type, these can be text, markdown and dataframe.
        # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
        # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
        # Prints the accuracy score.
        grid.cell("a", 1, 1, 2, 2).cell_text(function_accuracy_score)
        # Prints the f score.
        grid.cell("b", 2, 2, 2, 2).cell_text(mean_absolute_error_score)
        # Prints the precision score.
        grid.cell("c", 1, 1, 3, 3).cell_text(function_precision_score)
        # Prints the recall score.
        grid.cell("d", 2, 2, 3, 3).cell_text(mean_sqaured_error_score)

        # Prints the mean absolute error score.
        grid.cell("e", 1, 1, 4, 4).cell_text(function_f_one_score)
        # Prints the mean squared error score.
        grid.cell("f", 2, 2, 4, 4).cell_text(root_mean_sqaured_error_score)
        # Prints the root mean squared error score.
        grid.cell("g", 1, 1, 5, 5).cell_text(function_recall_score)
        # Display the total time taken for fitting the model.
        grid.cell("h", 2, 2, 6, 6).cell_text(prediction_string.format(total_prediction_time))
        # Display the total time taken for making the prediction.
        grid.cell("i", 1, 2, 6, 6).cell_text(fitting_string.format(total_fitting_time))
        # Display the confidence score for the predictions.
        grid.cell("j", 1, 1, 7, 7).cell_text("Confidence Score: " + str(random_forest_classifier.score(training_data_x, training_data_y)))

    # Sets prediction to the prediction_probabilities variable using argmax.
    prediction = np.argmax(prediction_probabilities, axis=-1)

    # Instantiates an instance of the confusion_heatmap_plotting_utility and passes the relevant informaiton to it.
    confusion_heatmap_plotting_utility(testing_data_y, prediction, emotion_keys)

    # Using the choose_data_type_for_algorithms from the initial choose data type if statement
    # we detect what the data type chosen is.
    # If text.
    if(choose_data_type_for_algorithms == "Text"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/text/probabilities/text_random_forest_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/text/probabilities/text_random_forest_classifier.pkl', 'rb') as f:
            random_forest_prediction_probabilities = pickle.load(f)
        # Using streamlits functionality, Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ text_random_forest_classifier.pkl_**")
    # Else if Audio.
    elif(choose_data_type_for_algorithms == "Audio"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_random_forest_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_random_forest_classifier.pkl', 'rb') as f:
            random_forest_prediction_probabilities = pickle.load(f)
        # Using streamlits functionality, Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ audio_random_forest_classifier.pkl_**")

    # Returns the prediction_probabilities, used when all algorithms have been ran
    # we use this to show a comparison.
    return prediction_probabilities

# Method responsible for training a support vector classifier.
def support_vector_classification_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms):
    # Instantiate variable support_vector_classification_classifier and set it equal to skleanrs LinearSVC functionality.
    support_vector_classification_classifier = LinearSVC()

    graphical_interface.write("SUPPORT")

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()
    # Using variable support_vector_classification_classifier we call sklearns fit functionality to fit the model and 
    # pass the relevant training data training_data_x, training_data_y.
    support_vector_classification_classifier.fit(training_data_x, training_data_y)

    # End the timer for the fitting of the function
    fitting_time_end = time.time()
    # Calculate the tootal time taken by deducting fitting_time_start from fitting_time_end.
    total_fitting_time = fitting_time_end-fitting_time_start

    # Start a timer for the prediction of the function.
    prediction_time_start = time.time()

    # Instantiate the variable prediction_probabilities and using the predic function from sklearns LinearSVC 
    # while passing the testing data set it equal to the outcome.
    # Making the actual prediction.
    prediction_probabilities = support_vector_classification_classifier.predict(testing_data_x)

    # End the timer for the predicting of the function
    prediction_time_end = time.time()
    # Calculate the tootal time taken by deducting prediction_time_start from prediction_time_end.
    total_prediction_time = prediction_time_end-prediction_time_start

    # Set to True as argmax is available for random forest 
    # --- See the notes in display_results_from_testing_utility ---.
    has_length_available = False
    # Specifies the averaging to be used by the metrics, I.e. f1 score, precision.
    averaging_to_be_used = 'micro'

    # Instanite the method display_results_from_testing_utility and set the passed variables equal
    # variables, thus we can print the informaiton to the screen.
    function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score = display_results_from_testing_utility(testing_data_y, prediction_probabilities, has_length_available, averaging_to_be_used)

    # Display that the training is copmplete to the user.
    graphical_interface.markdown("### Training Complete")

    # Creates a string which will be used in conjunction with total_fitting_time to display total time taken.
    fitting_string = 'Fitting took {0:.3f} seconds'
    # Creates a string which will be used in conjunction with prediction_string to display total time taken.
    prediction_string = 'Prediction took {0:.3f} seconds'

    # Checks if the fitting time is less than 1.
    if(total_fitting_time < 1):
        # If so then we replace the word seconds with milliseconds
        fitting_string = fitting_string.replace('seconds',' milliseconds')
    # Checks if the prediction time is less than 1.
    if(total_prediction_time < 1):
        # If so then we replace the word seconds with milliseconds
        prediction_string = prediction_string.replace('seconds',' milliseconds')
    
    # Using my grid API for generating a grid.
    with Grid("1 1") as grid:
        # Using the grid class we creatge a new grid element and pass it a cell type
        # the grid portion of this functionality takes a
        #
        # class this string is the identifier for the cell,
        # An int in position one which is the cells position width,
        # An int in position two which is the cells width,
        # An int in position three which is the position height,
        # An int in position four which is the cells height,

        # After this we pass a cell type, these can be text, markdown and dataframe.
        # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
        # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
        # Prints the accuracy score.
        grid.cell("a", 1, 1, 2, 2).cell_text(function_accuracy_score)
        # Prints the f score.
        grid.cell("b", 2, 2, 2, 2).cell_text(mean_absolute_error_score)
        # Prints the precision score.
        grid.cell("c", 1, 1, 3, 3).cell_text(function_precision_score)
        # Prints the recall score.
        grid.cell("d", 2, 2, 3, 3).cell_text(mean_sqaured_error_score)

        # Prints the mean absolute error score.
        grid.cell("e", 1, 1, 4, 4).cell_text(function_f_one_score)
        # Prints the mean squared error score.
        grid.cell("f", 2, 2, 4, 4).cell_text(root_mean_sqaured_error_score)
        # Prints the root mean squared error score.
        grid.cell("g", 1, 1, 5, 5).cell_text(function_recall_score)
        # Display the total time taken for fitting the model.
        grid.cell("h", 2, 2, 6, 6).cell_text(prediction_string.format(total_prediction_time))
        # Display the total time taken for making the prediction.
        grid.cell("i", 1, 2, 6, 6).cell_text(fitting_string.format(total_fitting_time))
 
        # Display the confidence score for the predictions.
        grid.cell("j", 1, 1, 7, 7).cell_text("Confidence Score: " + str(support_vector_classification_classifier.score(training_data_x, training_data_y)))

    # Instantiates an instance of the confusion_heatmap_plotting_utility and passes the relevant informaiton to it.
    confusion_heatmap_plotting_utility(testing_data_y, prediction_probabilities, emotion_keys)

    # Using the choose_data_type_for_algorithms from the initial choose data type if statement
    # we detect what the data type chosen is.
    # If text.
    if(choose_data_type_for_algorithms == "Text"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/text/probabilities/text_support_vector_classification_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/text/probabilities/text_support_vector_classification_classifier.pkl', 'rb') as f:
            support_vector_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ text_support_vector_classification_classifier.pkl_**")
    elif(choose_data_type_for_algorithms == "Audio"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_support_vector_classification_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_support_vector_classification_classifier.pkl', 'rb') as f:
            support_vector_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ audio_support_vector_classification_classifier.pkl_**")

    # Returns the prediction_probabilities, used when all algorithms have been ran
    # we use this to show a comparison.
    return prediction_probabilities

# Method responsible for training a linear regression multinominal naive bayes classifier.
def multinominal_naive_bayes_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms):
    # Instantiate variable multinominal_naive_bayes_classifier and set it equal to skleanrs MultinomialNB functionality.
    multinominal_naive_bayes_classifier = MultinomialNB()

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()

    # Using variable multinominal_naive_bayes_classifier we call sklearns fit functionality to fit the model and 
    # pass the relevant training data training_data_x, training_data_y.
    multinominal_naive_bayes_classifier.fit(training_data_x, training_data_y)

    # End the timer for the fitting of the function
    fitting_time_end = time.time()
    # Calculate the tootal time taken by deducting fitting_time_start from fitting_time_end.
    total_fitting_time = fitting_time_end-fitting_time_start

    # Start a timer for the prediction of the function.
    prediction_time_start = time.time()

    # Instantiate the variable prediction_probabilities and using the predic_proba function from sklearns MultinomialNB 
    # while passing the testing data set it equal to the outcome.
    # Making the actual prediction.
    prediction_probabilities = multinominal_naive_bayes_classifier.predict_proba(testing_data_x)

    # End the timer for the predicting of the function
    prediction_time_end = time.time()
    # Calculate the tootal time taken by deducting prediction_time_start from prediction_time_end.
    total_prediction_time = prediction_time_end-prediction_time_start
    # Using streamlits functionality, Display information to the user.
    graphical_interface.write('Table of prediction probabilities.')
    # Using streamlits functionality, Displays a table made up from the prediction proprobabilities to the user.
    graphical_interface.write(prediction_probabilities)

    # Set to True as argmax is available for random forest 
    # --- See the notes in display_results_from_testing_utility ---.
    has_length_available = True
    # Specifies the averaging to be used by the metrics, I.e. f1 score, precision.
    averaging_to_be_used = 'micro'

    # Instanite the method display_results_from_testing_utility and set the passed variables equal
    # variables, thus we can print the informaiton to the screen.
    function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score = display_results_from_testing_utility(testing_data_y, prediction_probabilities, has_length_available, averaging_to_be_used)
    # Using streamlits functionality, Display that the training is complete to the user.
    graphical_interface.markdown("### Training Complete")

    # Creates a string which will be used in conjunction with total_fitting_time to display total time taken.
    fitting_string = 'Fitting took {0:.3f} seconds'
    # Creates a string which will be used in conjunction with prediction_string to display total time taken.
    prediction_string = 'Prediction took {0:.3f} seconds'

    # Checks if the fitting time is less than 1.
    if(total_fitting_time < 1):
        # If so then we replace the word seconds with milliseconds
        fitting_string = fitting_string.replace('seconds',' milliseconds')
    # Checks if the prediction time is less than 1.
    if(total_prediction_time < 1):
        # If so then we replace the word seconds with milliseconds
        prediction_string = prediction_string.replace('seconds',' milliseconds')
    
    # Using my grid API for generating a grid.
    with Grid("1 1") as grid:
        # Using the grid class we creatge a new grid element and pass it a cell type
        # the grid portion of this functionality takes a
        #
        # class this string is the identifier for the cell,
        # An int in position one which is the cells position width,
        # An int in position two which is the cells width,
        # An int in position three which is the position height,
        # An int in position four which is the cells height,

        # After this we pass a cell type, these can be text, markdown and dataframe.
        # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
        # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
        # Prints the accuracy score.
        grid.cell("a", 1, 1, 2, 2).cell_text(function_accuracy_score)
        # Prints the f score.
        grid.cell("b", 2, 2, 2, 2).cell_text(mean_absolute_error_score)
        # Prints the precision score.
        grid.cell("c", 1, 1, 3, 3).cell_text(function_precision_score)
        # Prints the recall score.
        grid.cell("d", 2, 2, 3, 3).cell_text(mean_sqaured_error_score)
        # Prints the mean absolute error score.
        grid.cell("e", 1, 1, 4, 4).cell_text(function_f_one_score)
        # Prints the mean squared error score.
        grid.cell("f", 2, 2, 4, 4).cell_text(root_mean_sqaured_error_score)
        # Prints the root mean squared error score.
        grid.cell("g", 1, 1, 5, 5).cell_text(function_recall_score)
        # Display the total time taken for fitting the model.
        grid.cell("h", 2, 2, 6, 6).cell_text(prediction_string.format(total_prediction_time))
        # Display the total time taken for making the prediction.
        grid.cell("i", 1, 2, 6, 6).cell_text(fitting_string.format(total_fitting_time))

        # Display the confidence score for the predictions.
        grid.cell("j", 1, 1, 7, 7).cell_text("Confidence Score: " + str(multinominal_naive_bayes_classifier.score(training_data_x, training_data_y)))

    # Sets prediction to the prediction_probabilities variable using argmax.
    prediction = np.argmax(prediction_probabilities, axis=-1)

    # Instantiates an instance of the confusion_heatmap_plotting_utility and passes the relevant informaiton to it.
    confusion_heatmap_plotting_utility(testing_data_y, prediction, emotion_keys)

    # Using the choose_data_type_for_algorithms from the initial choose data type if statement
    # we detect what the data type chosen is.
    # If text.
    if(choose_data_type_for_algorithms == "Text"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/text/probabilities/text_multinominal_naive_bayes_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/text/probabilities/text_multinominal_naive_bayes_classifier.pkl', 'rb') as f:
            multi_nominal_naive_bayes_prediction_probabilities = pickle.load(f)
        # Using streamlits functionality, Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ text_multinominal_naive_bayes_classifier.pkl_**")
    elif(choose_data_type_for_algorithms == "Audio"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_multinominal_naive_bayes_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_multinominal_naive_bayes_classifier.pkl', 'rb') as f:
            multi_nominal_naive_bayes_prediction_probabilities = pickle.load(f)
        # Using streamlits functionality, Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ audio_multinominal_naive_bayes_classifier.pkl_**")
    
    # Returns the prediction_probabilities, used when all algorithms have been ran
    # we use this to show a comparison.
    return prediction_probabilities

# Method responsible for training a logistic regression classifier.
def logistic_regression_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms):
    # Instantiate variable logistic_regression_classifier and set it equal to skleanrs LogisticRegression functionality
    # while passing the relevant arguements, I.e set to handle multinomial loss, to handle multi nominal loss.
    # solver = Sets the algorithm used when in the optimsation problem.
    # multi class = The loss minimised is the multinomial loss fit across the entire probability distribution of the logisitc regression probability factor.
    # maximum iteration = Sets the absolute maximum iteration for the solvers to give coverage..
    logistic_regression_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()
    # Using variable logistic_regression_classifier we call sklearns fit functionality to fit the model and 
    # pass the relevant training data training_data_x, training_data_y.
    logistic_regression_classifier.fit(training_data_x, training_data_y)

    # End the timer for the fitting of the function
    fitting_time_end = time.time()
    # Calculate the tootal time taken by deducting fitting_time_start from fitting_time_end.
    total_fitting_time = fitting_time_end-fitting_time_start

    # Start a timer for the prediction of the function.
    prediction_time_start = time.time()

    # Instantiate the variable prediction_probabilities and using the predic_proba function from sklearns LogisticRegression 
    # while passing the testing data set it equal to the outcome.
    # Making the actual prediction.
    prediction_probabilities = logistic_regression_classifier.predict_proba(testing_data_x)

    # End the timer for the predicting of the function
    prediction_time_end = time.time()
    # Calculate the tootal time taken by deducting prediction_time_start from prediction_time_end.
    total_prediction_time = prediction_time_end-prediction_time_start

    # Display information to the user.
    graphical_interface.write('Table of prediction probabilities.')
    # Displays a table made up from the prediction proprobabilities to the user.
    graphical_interface.write(prediction_probabilities)

    # Set to True as argmax is available for random forest 
    # --- See the notes in display_results_from_testing_utility ---.
    has_length_available = True
    # Specifies the averaging to be used by the metrics, I.e. f1 score, precision.
    averaging_to_be_used = 'micro'

    # Instanite the method display_results_from_testing_utility and set the passed variables equal
    # variables, thus we can print the informaiton to the screen.
    function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score = display_results_from_testing_utility(testing_data_y, prediction_probabilities, has_length_available, averaging_to_be_used)
    # Display that the training is copmplete to the user.
    graphical_interface.markdown("### Training Complete")

    # Creates a string which will be used in conjunction with total_fitting_time to display total time taken.
    fitting_string = 'Fitting took {0:.3f} seconds'
    # Creates a string which will be used in conjunction with prediction_string to display total time taken.
    prediction_string = 'Prediction took {0:.3f} seconds'

    # Checks if the fitting time is less than 1.
    if(total_fitting_time < 1):
        # If so then we replace the word seconds with milliseconds
        fitting_string = fitting_string.replace('seconds',' milliseconds')
    # Checks if the prediction time is less than 1.
    if(total_prediction_time < 1):
        # If so then we replace the word seconds with milliseconds
        prediction_string = prediction_string.replace('seconds',' milliseconds')
    
    # Using my grid API for generating a grid.
    with Grid("1 1") as grid:
        # Using the grid class we creatge a new grid element and pass it a cell type
        # the grid portion of this functionality takes a
        #
        # class this string is the identifier for the cell,
        # An int in position one which is the cells position width,
        # An int in position two which is the cells width,
        # An int in position three which is the position height,
        # An int in position four which is the cells height,

        # After this we pass a cell type, these can be text, markdown and dataframe.
        # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
        # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
        # Prints the accuracy score.
        grid.cell("a", 1, 1, 2, 2).cell_text(function_accuracy_score)
        # Prints the f score.
        grid.cell("b", 2, 2, 2, 2).cell_text(mean_absolute_error_score)
        # Prints the precision score.
        grid.cell("c", 1, 1, 3, 3).cell_text(function_precision_score)
        # Prints the recall score.
        grid.cell("d", 2, 2, 3, 3).cell_text(mean_sqaured_error_score)
        # Prints the mean absolute error score.
        grid.cell("e", 1, 1, 4, 4).cell_text(function_f_one_score)
        # Prints the mean squared error score.
        grid.cell("f", 2, 2, 4, 4).cell_text(root_mean_sqaured_error_score)
        # Prints the root mean squared error score.
        grid.cell("g", 1, 1, 5, 5).cell_text(function_recall_score)
        # Display the total time taken for fitting the model.
        grid.cell("h", 2, 2, 6, 6).cell_text(prediction_string.format(total_prediction_time))
        # Display the total time taken for making the prediction.
        grid.cell("i", 1, 2, 6, 6).cell_text(fitting_string.format(total_fitting_time))

        # Display the confidence score for the predictions.
        grid.cell("j", 1, 1, 7, 7).cell_text("Confidence Score: " + str(logistic_regression_classifier.score(training_data_x, training_data_y)))

    # Sets prediction to the prediction_probabilities variable using argmax.
    prediction = np.argmax(prediction_probabilities, axis=-1)

    # Instantiates an instance of the confusion_heatmap_plotting_utility and passes the relevant informaiton to it.
    confusion_heatmap_plotting_utility(testing_data_y, prediction, emotion_keys)

    # Using the choose_data_type_for_algorithms from the initial choose data type if statement
    # we detect what the data type chosen is.
    # If text.
    if(choose_data_type_for_algorithms == "Text"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/text/probabilities/text_logistic_regression_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/text/probabilities/text_logistic_regression_classifier.pkl', 'rb') as f:
            logistic_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ text_logistic_regression_classifier.pkl_**")
    elif(choose_data_type_for_algorithms == "Audio"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_logistic_regression_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_logistic_regression_classifier.pkl', 'rb') as f:
            logistic_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ audio_logistic_regression_classifier.pkl_**")

    return prediction_probabilities

# Method responsible for training a multi perceptron neural network classifier.
def multi_perceptron_neural_network_method(training_data_x, training_data_y, testing_data_x, testing_data_y, emotion_dictionary, emotion_keys, choose_data_type_for_algorithms):
    # Instantiate variable multi_perceptron_neural_network_method and set it equal to sklearns MLPClassifier and pass the relevant arguements, See below.

    # hidden_layer_sizes = Represents the ith number of actual neurons present within the hidden layer.
    # activation = Sets the activation method, in this case rectified linear unti.
    # solver = Sets the resolver
    # alpha = Regularization parameter, regularize the inputs.
    # batch_size = Size of minibatches for stochastic optimizers.
    # learning_rate = Sets the learning rate, I.e. adaptive sets this to learning rate constant to the learning rates init.
    # learning_rate_init = The initial learning rate of the neural network.
    # power_t = The exponent of the learning rate inversed.
    # maximum_iteration = Sets the maximum iterations for the neural network.
    # shuffle = Set to true this will shufle the smaples each iteration.
    # random_state = Generates a random number.
    # tol = Sets the tolerance of the optmization, only when the loss is not improving.
    # verbose = Sets the verbosity, I.e. prints the progress, learning rate amongst other information to the terminal.
    # warm_start = Set to True, this will reuse the solution of the previous fit as initialization
    # momentum = Sets the gradient decent update per iteration.
    # nesterovs_momentum = Sets wether or not to use the nesterovs.
    # early_stopping = If true a portion of the training data will set aside for validation, I.e. a validation set.
    # beta_1 = Sets the decay rates rates, exponential, of the second vector (adam specific).
    # beta_2 = Sets the decay rates rates, exponential, of the second moment (adam specific).
    # epsilon = Sets the numerical stability value (adam specific).
    multi_perceptron_neural_network_method = MLPClassifier(hidden_layer_sizes=(650, ), activation='relu', solver='adam', alpha=0.0001,
                                                        batch_size='auto', learning_rate='adaptive', learning_rate_init=0.01,
                                                        power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001,
                                                        verbose=False, warm_start=True, momentum=0.8, nesterovs_momentum=True,
                                                        early_stopping=False, beta_1=0.9, beta_2=0.999,
                                                        epsilon=1e-08)

    # Start a timer for the fitting of the function.
    fitting_time_start = time.time()
    # Using the fit function from sklearns MLPClassifier 
    # while passing the training_data_x, training_data_y variables we fit the model.
    # Making the actual fitting.
    multi_perceptron_neural_network_method.fit(training_data_x, training_data_y)
    # End the timer for the fitting of the function
    fitting_time_end = time.time()
    # Calculate the tootal time taken by deducting fitting_time_start from fitting_time_end.
    total_fitting_time = fitting_time_end-fitting_time_start
    # Start a timer for the prediction of the function.
    prediction_time_start = time.time()

    # Instantiate the variable prediction_probabilities and using the predic_proba function from sklearns MLPClassifier 
    # while passing the testing data set it equal to the outcome.
    # Making the actual prediction.
    prediction_probabilities = multi_perceptron_neural_network_method.predict_proba(testing_data_x)
    # End the timer for the predicting of the function
    prediction_time_end = time.time()
    # Calculate the tootal time taken by deducting prediction_time_start from prediction_time_end.
    total_prediction_time = prediction_time_end-prediction_time_start
    # Display information to the user.
    graphical_interface.write('Table of prediction probabilities.')
    # Displays a table made up from the prediction proprobabilities to the user.
    graphical_interface.write(prediction_probabilities)

    # Set to True as argmax is available for random forest 
    # --- See the notes in display_results_from_testing_utility ---.
    has_length_available = True

    # Specifies the averaging to be used by the metrics, I.e. f1 score, precision.
    averaging_to_be_used = 'macro'

    # Instanite the method display_results_from_testing_utility and set the passed variables equal
    # variables, thus we can print the informaiton to the screen.
    function_accuracy_score, function_f_one_score, function_precision_score, function_recall_score, mean_absolute_error_score, mean_sqaured_error_score, root_mean_sqaured_error_score = display_results_from_testing_utility(testing_data_y, prediction_probabilities, has_length_available, averaging_to_be_used)
    # Display that the training is copmplete to the user.
    graphical_interface.markdown("### Training Complete")
    
    # Creates a string which will be used in conjunction with total_fitting_time to display total time taken.
    fitting_string = 'Fitting took {0:.3f} seconds'
    # Creates a string which will be used in conjunction with prediction_string to display total time taken.
    prediction_string = 'Prediction took {0:.3f} seconds'

    # Checks if the fitting time is less than 1.
    if(total_fitting_time < 1):
        # If so then we replace the word seconds with milliseconds
        fitting_string = fitting_string.replace('seconds',' milliseconds')
    # Checks if the prediction time is less than 1.
    if(total_prediction_time < 1):
        # If so then we replace the word seconds with milliseconds
        prediction_string = prediction_string.replace('seconds',' milliseconds')
    
    # Using my grid API for generating a grid.
    with Grid("1 1") as grid:
        # Using the grid class we creatge a new grid element and pass it a cell type
        # the grid portion of this functionality takes a
        #
        # class this string is the identifier for the cell,
        # An int in position one which is the cells position width,
        # An int in position two which is the cells width,
        # An int in position three which is the position height,
        # An int in position four which is the cells height,

        # After this we pass a cell type, these can be text, markdown and dataframe.
        # If the cell is text we pass the text directly to the grid/cell, I.e. grid.cell("b", 2, 3, 2, 3).text("Text Example")
        # If the cell is a dataframe, we the dataframe directly to the cell, I.e. grid.cell("d", 3, 4, 1, 2).dataframe(dataframe_method())
        # Prints the accuracy score.
        grid.cell("a", 1, 1, 2, 2).cell_text(function_accuracy_score)
        # Prints the f score.
        grid.cell("b", 2, 2, 2, 2).cell_text(mean_absolute_error_score)
        # Prints the precision score.
        grid.cell("c", 1, 1, 3, 3).cell_text(function_precision_score)
        # Prints the recall score.
        grid.cell("d", 2, 2, 3, 3).cell_text(mean_sqaured_error_score)
        # Prints the mean absolute error score.
        grid.cell("e", 1, 1, 4, 4).cell_text(function_f_one_score)
        # Prints the mean squared error score.
        grid.cell("f", 2, 2, 4, 4).cell_text(root_mean_sqaured_error_score)
        # Prints the root mean squared error score.
        grid.cell("g", 1, 1, 5, 5).cell_text(function_recall_score)
        # Display the total time taken for fitting the model.
        grid.cell("h", 2, 2, 6, 6).cell_text(prediction_string.format(total_prediction_time))
        # Display the total time taken for making the prediction.
        grid.cell("i", 1, 2, 6, 6).cell_text(fitting_string.format(total_fitting_time))

        # Display the confidence score for the predictions.
        grid.cell("j", 1, 1, 7, 7).cell_text("Confidence Score: " + str(multi_perceptron_neural_network_method.score(training_data_x, training_data_y)))

    # Sets prediction to the prediction_probabilities variable using argmax.
    prediction = np.argmax(prediction_probabilities, axis=-1)

    # Instantiates an instance of the confusion_heatmap_plotting_utility and passes the relevant informaiton to it.
    confusion_heatmap_plotting_utility(testing_data_y, prediction, emotion_keys)
            
    # Using the choose_data_type_for_algorithms from the initial choose data type if statement
    # we detect what the data type chosen is.
    # If text.
    if(choose_data_type_for_algorithms == "Text"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/text/probabilities/text_multi_layer_perceptron_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/text/probabilities/text_multi_layer_perceptron_classifier.pkl', 'rb') as f:
            multi_layer_perceptron_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ text_multi_layer_perceptron_classifier.pkl_**")
    elif(choose_data_type_for_algorithms == "Audio"):
        # Saves the prediction probabilities.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_multi_layer_perceptron_classifier.pkl', 'wb') as f:
            pickle.dump(prediction_probabilities, f)

        # Loads the prediction probabilities from a pickle file.
        with open('utilities/data/proccessed_data/audio/probabilities/audio_multi_layer_perceptron_classifier.pkl', 'rb') as f:
            multi_layer_perceptron_prediction_probabilities = pickle.load(f)
        # Prints the fact that the model has been saved to the user.
        graphical_interface.markdown("#### Model Has Been Saved As, **_ audio_multi_layer_perceptron_classifier.pkl_**")
    # Return the prediction_probabilities.
    return prediction_probabilities

# Main method.
def main():
    # Using streamlits functionality, user action selection section.
    choose_action_type = graphical_interface.sidebar.selectbox("Choose a data type",
                                                             ["NONE",
                                                              "------------------------------------------",
                                                              "1: Pre Process Textual Data",
                                                              "2: Pre Process Auditory Data",
                                                              "3: Process And Prepare All Data",
                                                              "------------------------------------------",
                                                              "4: Select And Train A Model",
                                                              "------------------------------------------"])

    # If none is selected.
    if(choose_action_type == 'NONE'):
        # Display information to the user.
        graphical_interface.write('Please select a learning method to open more options')
        # Instantiate the no_method_selected is selected.
        no_method_selected()
    # Elif the user selectes 1, Extract Emotion Labels From Text.
    elif(choose_action_type == "1: Pre Process Textual Data"):
        # Call method extract_emotion_labels_method.
        pre_process_textual_data()
    # Elif the user selectes 2, Build Vectors From Audio.
    elif(choose_action_type == "2: Pre Process Auditory Data"):
        # Call method build_vectors_method.
        pre_process_auditory_data()
    # Elif the user selectes 3, Extract Features From Audio.
    elif(choose_action_type == "3: Process And Prepare All Data"):
        # Call method extract_features_method.
        process_and_prepare_data_method()
    # Elif the user selectes 5, Select And Train A Model.
    elif(choose_action_type == "4: Select And Train A Model"):
        # Instantiate the audio_data_type_selected_method is selected.
        data_and_model_selection_method()

# If statement to check for main.
if __name__ == "__main__":
    main()
