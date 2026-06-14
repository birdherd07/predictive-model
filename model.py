import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.compose import ColumnTransformer
from tensorflow.keras import models, layers
import numpy as np
from scipy.spatial import KDTree
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

# A machine learning model which takes test data files containing only population and location data 
# and produces predictions for the vote proportions in each block
#the input: an arbitrary set of blocks (any state) in test format - population and location and total votes
#output: predictions for vote proportions in each block. blocks around each one are relevant.


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

#Maps FCN output tensor back to original county IDs
def extract_predictions(fcn_output, county_mappings):
    results = {}

    with torch.no_grad():
        for mapping in county_mappings:
            id = mapping['id']
            x = mapping['grid_x']
            y = mapping['grid_y']

            pixel_prediction = fcn_output[:, y, x].cpu().numpy()

            results[id] = pixel_prediction

    #dictionary mapping {county ID: prediction vector[2]} <- D% and R%
    return results


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

normalizer = StandardScaler()
training_data = []
training_labels = []
model = None

#Standardize population column of training data using z-score normalization (all data in single ndarray)
def normalize_training_np(trainingData: np.ndarray, populationCol: int, normalizer: StandardScaler):
    #transformer = ColumnTransformer(transformers = [('scaler', normalizer, [populationCol])], remainder='passthrough')
    
    trainingData[:, [populationCol]] = normalizer.fit_transform(trainingData[:, [populationCol]])
    #print(f"{normalizer.mean_}, {normalizer.scale_}")

#Standardize population column of training data using z-score normalization (collection of state 2d ndarrays)
def normalize_training_list_np(trainingListData, populationCol: int, normalizer: StandardScaler):
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
    

#Create a fully convolutional model.
def create_fcn(classes = 2):
    print("Creating a new model...")
    #fully convolutional network: output for each block in the map
    inputLayer = layers.Input(shape=(None, None, 3))

    #encoder: blocks of convolution layers
    enc1 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputLayer)
    enc1 = layers.MaxPooling2D(pool_size=(2,2))(enc1)

    enc2 = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(enc1)
    enc2 = layers.Dropout(0.15)(enc2)
    enc2 = layers.MaxPooling2D(pool_size=(2,2))(enc2)

    enc3 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(enc2)
    enc3 = layers.MaxPooling2D(pool_size=(2,2))(enc3)

    #decoder: blocks of transpose layers
    dec1 = layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same')(enc3)
    dec2 = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same')(dec1)
    dec3 = layers.Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same')(dec2)
    
    #output layer: 1, 1 convolution layer with the 2 output classes
    #leave this kernel initializer default for softmax (glorot_uniform)
    outputLayer = layers.Conv2D(classes, kernel_size=(1,1), activation='softmax', padding='same')(dec3)
    model = models.Model(inputs=[inputLayer], outputs=[outputLayer])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

#Convert latitude and longitude center point to 2D grid
def convert_location(data, buffer=1.05):
    #Assuming data is of format ID LON LAT POP VOTES R% D%
    print("Converting counties to grid...")
        #get max and min lat and long values to convert to grid

    #Use density of counties to determine grid size
    coords = data[:, [1,2]]
    nn_tree = KDTree(coords)
    distances, _ = nn_tree.query(coords, k=2)
    min_distance = np.min(distances[:, 1])
    pixel_size = min_distance * 0.5

    lon_min = data[:, 1].min()
    lon_max = data[:, 1].max()
    lat_min = data[:, 2].min()
    lat_max = data[:, 2].max()

    grid_width = int(np.ceil((lon_max - lon_min) / pixel_size * buffer))
    grid_height = int(np.ceil((lat_max - lat_min) / pixel_size * buffer))

    #Reduce grid size if too large for memory
    GRID_MAX = 2048
    if grid_width > GRID_MAX or grid_height > GRID_MAX:
        print(f"Grid size {grid_width} x {grid_height} being reduced for safety: {GRID_MAX} x {GRID_MAX}. May cause collisions")
        scale = GRID_MAX / max(grid_width, grid_height)
        grid_width = int(grid_width * scale)
        grid_height = int(grid_height * scale)
        pixel_size = min_distance * 0.5

    county_mappings = []

    #Channel 0: population. Channel 1: presence/absence of a county
    grid = np.zeros((2, grid_height, grid_width), dtype=np.float32)
    labels_grid = np.zeros(2, grid_height, grid_width, dtype=np.float32)

    for row in data:
        # Map to [0, width-1] and [0, height-1]
        x = int(np.round(((row[1] - lon_min) / (lon_max - lon_min)) * (grid_width - 1)))
        y = int(np.round(((row[2] - lat_min) / (lat_max - lat_min)) * (grid_height - 1)))

        #Make north at top of grid
        invert_y = (grid_height - 1) - y

        #Place population, presence at point in grid
        grid[0, invert_y, x] += row[3]
        grid[1, invert_y, x] = 1.0

        #place R%, D% at point 
        labels_grid[0, invert_y, x] = row[5]
        labels_grid[0, invert_y, x] = row[6]

        #For retrieving counties by ID from FCN output
        county_mappings.append({
            'c_id': data[0],
            'grid_x': x,
            'grid_y': invert_y
        })

    metadata = {
    "grid_shape": (grid_height, grid_width),
    "pixel_size": pixel_size,
    "lon_range": (lon_min, lon_max),
    "lat_range": (lat_min, lat_max)
    }
    
    return grid, county_mappings, metadata, labels_grid

#divide grids into groups of 3 for batching
def batch_grids(grids):
    print("Dividing data into batches...")
    sizes = np.array([x.size for x in grids])
    sort_order = np.argsort(sizes)
    sorted_grids = grids[sort_order]

    batches = []

    for i in sorted_grids:
        batch.append()

    return sorted_grids


#pad grids in this batch to the largest size and update mappings for batch index
def pad_batch(grids, mappings):
    #pad grids to the largest size grid in this batch
    max_h = max(g.shape[1] for g in grids)
    max_w = max(g.shape[2] for g in grids)

    padded_grids = []
    padded_mappings = []

    #pad the right and bottom with 0s
    for batch_id, (grid, mapping) in enumerate(zip(grids, mappings)):
        c, h, w = grid.shape
        pad_h = max_h - h
        pad_w = max_w - w

        padded_grid = torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0)
        padded_grids.append(padded_grid)

        for m in mapping:
            padded_mappings.append({
                'id': m['id'],
                'b_id': batch_id,
                'x': m['x'],
                'y': m['y']
            })

    return torch.stack(padded_grids), padded_mappings

class CountyDataset(Dataset):
    def __init__(self, data_list, training: bool):
        #data_list: a 3d ndarr.
        #each 2d ndarr is a group of counties of form of format 
        #ID LON LAT POP VOTES R% D% (training = True)
        #or 
        #ID LON LAT POP (testing / training = False)
        self.data = data_list
        self.training = training
        self.grid_sizes = []
    
    def __len__(self):
        return len(self.data)
    
    #get transformed data and labels
    def __getitem__(self, id):
        #idx = index of list of data (a 2d ndarr)
        item = self.data[id]

        #Turn training data into a grid
        data_grid, county_mappings, metadata = self.rasterize_data(item)

        #Make matching training label grid
        h, w = metadata["grid_shape"]
        labels_grid = np.zeros((2, h, w), dtype=np.float32)

        for i, mapping in enumerate(county_mappings):
            x = mapping["grid_x"]
            y = mapping["grid_y"]

            labels_grid[0, y, x] = item[i][5]
            labels_grid[1, y, x] = item[i][6]

        input_tensor = torch.from_numpy(data_grid)
        target_tensor = torch.from_numpy(labels_grid)

        return input_tensor, target_tensor, county_mappings

    #Use latitude and longitude to convert list of points to a 2D grid
    def rasterize_data(data, buffer=1.05):
        print("Converting counties to grid...")

        #Use density of counties to determine grid size
        coords = data[:, [1,2]]
        nn_tree = KDTree(coords)
        distances, _ = nn_tree.query(coords, k=2)
        min_distance = np.min(distances[:, 1])
        pixel_size = min_distance * 0.5

        lon_min = data[:, 1].min()
        lon_max = data[:, 1].max()
        lat_min = data[:, 2].min()
        lat_max = data[:, 2].max()

        grid_width = int(np.ceil((lon_max - lon_min) / pixel_size * buffer))
        grid_height = int(np.ceil((lat_max - lat_min) / pixel_size * buffer))

        #Reduce grid size if too large for memory
        GRID_MAX = 2048
        if grid_width > GRID_MAX or grid_height > GRID_MAX:
            print(f"Grid size {grid_width} x {grid_height} being reduced for safety: {GRID_MAX} x {GRID_MAX}. May cause collisions")
            scale = GRID_MAX / max(grid_width, grid_height)
            grid_width = int(grid_width * scale)
            grid_height = int(grid_height * scale)
            pixel_size = min_distance * 0.5

        county_mappings = []

        #Channel 0: population. Channel 1: presence/absence of a county
        grid = np.zeros((2, grid_height, grid_width), dtype=np.float32)

        for row in data:
            # Map to [0, width-1] and [0, height-1]
            x = int(np.round(((row[1] - lon_min) / (lon_max - lon_min)) * (grid_width - 1)))
            y = int(np.round(((row[2] - lat_min) / (lat_max - lat_min)) * (grid_height - 1)))

            #Make north at top of grid
            invert_y = (grid_height - 1) - y

            #Place population, presence at point in grid
            grid[0, invert_y, x] += row[3]
            grid[1, invert_y, x] = 1.0

            #For retrieving counties by ID from FCN output
            county_mappings.append({
                'c_id': data[0],
                'grid_x': x,
                'grid_y': invert_y
            })

        metadata = {
        "grid_shape": (grid_height, grid_width),
        "pixel_size": pixel_size,
        "lon_range": (lon_min, lon_max),
        "lat_range": (lat_min, lat_max)
        }
        
        return grid, county_mappings, metadata

class SizeBasedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, grid_sizes):
        super().__init(sampler, batch_size, drop_last)
        self.grid_sizes = np.array(grid_sizes)

    #group similarly sized grids into batches of given size
    def __iter__(self):
        sorted_indices = np.argsort(self.grid_sizes)
        batch = []
        for id in sorted_indices:
            batch.append(id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    #pad grids in given batch to the largest size and update mappings for batch index
    def pad_batch(batch):
        data_grids = [item[0] for item in batch]
        label_grids = [item[1] for item in batch]
        mappings = [item[2] for item in batch]

        #pad grids to the largest size grid in this batch
        max_h = max(g.shape[1] for g in data_grids)
        max_w = max(g.shape[2] for g in data_grids)

        padded_grids = []
        padded_labels = []
        padded_mappings = []

        #pad the right and bottom with 0s
        for batch_id, (grid, label, mapping) in enumerate(zip(data_grids, label_grids, mappings)):
            pad_h = max_h - grid.shape[1]
            pad_w = max_w - grid.shape[2]

            padded_grids.append(torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0))
            padded_labels.append(torch.nn.functional.pad(label, (0, pad_w, 0, pad_h), value=0))

            for m in mapping:
                padded_mappings.append({
                    'id': m['id'],
                    'batch_id': batch_id,
                    'x': m['x'],
                    'y': m['y']
                })

        return torch.stack(padded_grids), torch.stack(padded_labels), padded_mappings

#Load training data and train a new model.
def train_model(model):
    global training_data, training_labels
    folder = input("Enter the location of the folder containing training data.\n")
    training_location = os.path.join(folder, "*.csv")
    training_files = glob.glob(training_location)

    #print(f"Found training files: {training_files}")
    if not training_files:
        print("No training files found.")
        return
    else:
        print(f"Found {len(training_files)} training files.\n")

    # training_headers = ["ID", "LON", "LAT", "POP", "VOTES", "REP %", "DEM %"]
    # data_headers = training_headers[:4]
    # label_headers = training_headers[-3:]

    # training_data = []
    # training_labels = []

    for file in training_files:
        #separate the training data from the labels
        #data = pd.read_csv(file, nrows=50, header=None, names=training_headers, usecols=data_headers)
        data = pd.read_csv(file, nrows=100, header=None, usecols=range(4)).to_numpy()
        #change these cols for testing files. add column 4 if predicting # votes
        labels = pd.read_csv(file, nrows=100, header=None, usecols=[5, 6])
        #print(data.head())
        # print("\n")
        # print(data[:5])
        # print(labels.head())
        training_data.append(data)
        training_labels.append(labels)

    #print(training_data[0][:5])
    # data = pd.read_csv(training_files[0], header=None, names=training_headers, usecols=data_headers)
    # training_data.append(data)
    # print(data.head())

    print("Processing training data...")
    #normalize_training(training_data[0], "POP")
    #normalize_training_list(training_data, "POP")
    #normalize_training_np(training_data[0], 3)

    training_data = np.array(training_data)
    training_labels = np.array(training_labels)

    normalize_training_list_np(training_data, 3, normalizer)

    print(f"{normalizer.mean_}, {normalizer.scale_}")
    
    dataset = CountyDataset(training_data, True)

    base_sampler = SequentialSampler(dataset)
    custom_batch_sampler = SizeBasedBatchSampler(
        base_sampler,
        batch_size=3,
        drop_last=False,
        grid_sizes=dataset.grid_sizes
    )

    dataloader = DataLoader(
        dataset, batch_sampler = custom_batch_sampler,
        collate_fn = pad_batch
    )

    for padded_grids, mappings in dataloader:
        print(f"Batch tensor shape: {padded_grids.shape}")


    #training_grid, county_mappings, metadata, labels_grid  = zip(*[convert_location(training_data[i]) for i in training_data])

    #training_grid, county_mappings, metadata = list(training_grid), list(county_mappings), list(metadata)

    #group maps in batches for training or pad them all to the largest size
    print("Batching data...")
    #sort by size
    #divide into size-3 batches
    #pad batches to same size

    #how to format labels???

    print("Training the model...")

    #model.fit(training_grid, training_labels,)
    


    #separate the labels from the data

    #reserve an amount of the training data + labels for validation
    #run training


keep_running = True
print("- Jerry Mandarin -")
while keep_running:
    train = input("\nWould you like to train a new model? [Y/N]\n")

    if train.upper() == 'Y':
        model = create_fcn()
        train_model(model)


        #print(training_data[0][:5])
    
    # testing = input("\nWould you like to run a model? [Y/N]\n")
    # if testing.upper() == 'Y':
    #     test_model()

    quit = input("Would you like to quit? [Y/N]\n")
    if quit.upper() == 'Y':
        keep_running = False
