import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer
from tensorflow.keras import models, layers
import numpy as np

# A machine learning model which takes test data files containing only population and location data 
# and produces predictions for the vote proportions in each block
#the input: an arbitrary set of blocks (any state) in test format - population and location and total votes
#output: predictions for vote proportions in each block. blocks around each one are relevant.

normalize = StandardScaler()
training_data = []
training_labels = []
model = None

#Standardize population column of training data using z-score normalization (all data in single ndarray)
def normalize_training_np(trainingData: np.ndarray, populationCol: int, normalizer: StandardScaler):
    #transformer = ColumnTransformer(transformers = [('scaler', normalizer, [populationCol])], remainder='passthrough')
    
    trainingData[:, [populationCol]] = normalizer.fit_transform(trainingData[:, [populationCol]])
    #print(f"{normalizer.mean_}, {normalizer.scale_}")


#Standardize population column of training data using z-score normalization (list of ndarrays)
def normalize_training_list_np(trainingListData: list[np.ndarray], populationCol: int, normalizer: StandardScaler):
    #Calculate the mean and standard deviation over all training ndarrays
    for trainingData in trainingListData:
        normalizer.partial_fit(trainingData[:, [populationCol]])

    #Transform each frame using the aggregated values
    for trainingData in trainingListData:
        trainingData[:, [populationCol]] = normalizer.transform(trainingData[:, [populationCol]])  
    #print(f"{normalizer.mean_}, {normalizer.scale_}")


#Use the normalizer from training to scale the population column of test data.
def normalize_testing_np(testData: np.ndarray, populationCol: int, normalizer: StandardScaler):
    testData[:, [populationCol]] = normalizer.transform(testData[:, [populationCol]])


#Standardize population column of training data using z-score normalization (all data in single dataframe)
def normalize_training(trainingData: pd.DataFrame, populationCol: str):
    trainingData[populationCol] = normalizer.fit_transform(trainingData[[populationCol]])
    #print(f"{normalizer.mean_}, {normalizer.scale_}")

#Standardize population column of training data using z-score normalization (list of dataframes)
def normalize_training_list(trainingListData: list[pd.DataFrame], populationCol: str):
    #Calculate the mean and standard deviation over all training dataframes
    for trainingData in trainingListData:
        normalizer.partial_fit(trainingData[[populationCol]])

    #Transform each frame using the aggregated values
    for trainingData in trainingListData:
        trainingData[populationCol] = normalizer.transform(trainingData[[populationCol]])  
    #print(f"{normalizer.mean_}, {normalizer.scale_}")

#Use the normalizer from training to scale the population column of test data.
def normalize_testing(testData: pd.DataFrame, populationCol: str):
    normalizer.transform(testData[[populationCol]])


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
    global model
    model = models.Sequential(name="jerry_mandarin")
    #variable size input
    model.add(layers.Input(shape=(None, None, 3)))
    #larger kernel for high level structures in first layer only
    model.add(layers.Conv2D(filters=32, kernel_size=(7,7), activation='relu'))
    #double filters each time layer deepens
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPooling2D(pooling_size=(2,2)))
    #in place of flattening due to unknown input size
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.15))
    #maybe replace this with actual classifier names
    model.add(layers.Dense(2, activation='softmax'))
    #if labels are 1-hot encoded ex 1.0, 0.0 then use categorical_crossentropy, else sparse for integer labels ex 0, 1
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


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
    data_headers = training_headers[:4]
    label_headers = training_headers[-3:]

    # training_data = []
    # training_labels = []

    for file in training_files:
        #separate the training data from the labels
        #data = pd.read_csv(file, nrows=50, header=None, names=training_headers, usecols=data_headers)
        data = pd.read_csv(file, nrows=50, header=None, usecols=range(4)).to_numpy()
        # labels = pd.read_csv(file, header=None, names=training_headers, usecols=label_headers)
        #print(data.head())
        print("\n")
        print(data[:5])
        # print(labels.head())
        training_data.append(data)
    #   training_labels.append(labels)

    #print(training_data[0][:5])
    # data = pd.read_csv(training_files[0], header=None, names=training_headers, usecols=data_headers)
    # training_data.append(data)
    # print(data.head())

    #create_model()

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
        #normalize_training(training_data[0], "POP")
        #normalize_training_list(training_data, "POP")
        #normalize_training_np(training_data[0], 3)
        normalize_training_list_np(training_data, 3, normalize)
        print(f"{normalize.mean_}, {normalize.scale_}")
        for x in training_data:
            #print(x.head())
            print("\n")
            print(x[:5])

        #print(training_data[0][:5])
    
    # testing = input("\nWould you like to run a model? [Y/N]\n")
    # if testing.upper() == 'Y':
    #     test_model()

    quit = input("Would you like to quit? [Y/N]\n")
    if quit.upper() == 'Y':
        keep_running = False
