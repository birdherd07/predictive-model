import os
import glob
import pandas as pd

# A machine learning model which takes test data files containing only population and location data 
# and produces predictions for the vote proportions in each block
#the input: an arbitrary set of blocks (any state) in test format - population and location and total votes
#output: predictions for vote proportions in each block. blocks around each one are relevant.


#Load test data and use a trained model to make predictions.
def test_model():
    if not glob.glob("*.keras"):
        print("Warning: No trained models found.")
        return
    
    folder = input("Enter the location of the folder containing testing data.\n")
    testing_location = os.path.join(folder, "*.csv")
    testing_files = glob.glob(testing_location)

    #print(f"Found test files: {testing_files}")
    if not testing_files:
        print("No testing files found.")
        return
    else:
        print(f"Found {len(testing_files)} testing files.\n")

    testing_headers = ["ID", "LON", "LAT", "POP", "VOTES"]
    for file in testing_files:
        df = pd.read_csv(file, names=testing_headers)
        print(df.head())
    

#Create a new model.
def create_model():
    print("Creating the model...")
    #massive NN, small NN or combo?


#Load training data and train a new model.
def train_model():
    folder = input("Enter the location of the folder containing training data.\n")
    training_location = os.path.join(folder, "*.csv")
    training_files = glob.glob(training_location)

    #print(f"Found training files: {training_files}")
    if not training_files:
        print("No training files found.")
        return
    else:
        print(f"Found {len(training_files)} training files.\n")

    training_headers = ["ID", "LON", "LAT", "POP", "VOTES", "REP %", "DEM %"]
    data_headers = training_headers[:5]
    label_headers = training_headers[-2:]

    training_data = []
    training_labels = []

    for file in training_files:
        #separate the training data from the labels
        data = pd.read_csv(file, header=None, names=training_headers, usecols=data_headers)
        labels = pd.read_csv(file, header=None, names=training_headers, usecols=label_headers)
        print(data.head())
        print(labels.head())

        #training_data.append(data)
        #training_labels.append(labels)

    create_model()

    print("Training the model...")
    #reshape data for model type chosen

    #separate the labels from the data

    #reserve an amount of the training data + labels for validation
    #run training


keep_running = True
print("- Jerry Mandarin -")
while keep_running:
    train = input("\nWould you like to train a new model? [Y/N]\n")

    if train.upper() == 'Y':
        train_model()
    
    testing = input("\nWould you like to run a model? [Y/N]\n")
    if testing.upper() == 'Y':
        test_model()

    quit = input("Would you like to quit? [Y/N]\n")
    if quit.upper() == 'Y':
        keep_running = False
