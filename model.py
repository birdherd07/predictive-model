import os
import glob
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
#from sklearn.compose import ColumnTransformer
import tensorflow as tf
import keras
from tensorflow.keras import models, layers
import numpy as np
#from scipy.spatial import KDTree
from scipy.spatial import cKDTree as KDTree
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
import joblib

# A machine learning model which takes test data files containing only population and location data 
# and produces predictions for the vote proportions in each block

#Use the normalizer from training to scale the population column of test data.
def normalize_testing(testData: pd.DataFrame, populationCol: str):
    normalizer.transform(testData[[populationCol]])

#Maps FCN output tensor back to original county IDs
def extract_predictions(fcn_output, county_mappings):
    results = {}

    with torch.no_grad():
        for mapping in county_mappings:
            id = mapping['id']
            x = mapping['x']
            y = mapping['y']

            pixel_prediction = fcn_output[:, y, x].cpu().numpy()

            results[id] = pixel_prediction

    #dictionary mapping {county ID: prediction vector[2]} <- D% and R%
    return results


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
    if not model:
        model_loc = input("Enter the location of the model file.\n")
        try:
            model = models.load_model(model_loc)
        except:
            print(f"No model found at a {model_loc}.")
            return
        
    try:
        check_is_fitted(normalizer)
    except:
        scaler_name = input("Enter the name of the scaler bin file.\n")
        try:
            normalizer = joblib.load(scaler_name)
        except:
            print("No scaler file found in current directory.")
            return

    dataset, length = load_data()

    #run model
    for i in range(length):
        #3, h, w
        input_tensor, county_mappings = dataset[i]
        model_input = tf.expand_dims(input_tensor, axis=0)

        output_map = model(model_input, training=False)
        pct_map = tf.keras.activations.softmax(output_map, axis=1)

        #3, h, w
        pct_map = tf.squeeze(pct_map, axis=0)
        results = extract_predictions(pct_map, county_mappings)

        print(f"Predictions will be written to file as: predictions{i+1}.csv\n")

        #Output list of ids and predictions to csv
        #format: id, R%, D%
        with open(f'predictions{i+1}.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for key, value in results:
                writer.writerow([key, value])




#Create a fully convolutional model. third channel is the remainder of 100 - (r% + d%) for softmax
def create_fcn(input_channels=2, classes=3):
    print("Creating a new model...")

    #fully convolutional network: output for each block in the map.
    #shape: channels (presence, population), none (any), none (any)
    inputs = layers.Input(shape=(input_channels, None, None))
    
    # convolutional block 1: neighbors in a 3x3 grid. batchnormalization layer for batches > 1
    x = layers.Conv2D(32, kernel_size=3, padding='same', data_format='channels_first')(inputs)
    #x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    
    # convolutional block 2: dilation to wider area
    x = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=2, data_format='channels_first')(x)
    #x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)

    # convolutional block 3
    x = layers.Conv2D(64, kernel_size=3, padding='same', data_format='channels_first')(x)
    #x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dropout(rate=0.2)(x)

    # Final layer mapping to our 2 output percentage channels
    outputs = layers.Conv2D(classes, kernel_size=1, padding='same', data_format='channels_first')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

#---------------------------------------------------------------------------------------------------------------
#Custom dataset class that transforms list data into 2d rasters for FCN
class CountyDataset(Dataset):
    def __init__(self, data_list, training: bool):
        #data_list: a 3d ndarr.
        #each 2d ndarr is a group of counties of form of format 
        #ID LON LAT POP VOTES R% D% (training = True)
        #or 
        #ID LON LAT POP (testing. training = False)
        #todo: implement testing version

        self.data = data_list
        self.training = training
        self.grid_sizes = []
        self.grid_info = []
        for item in self.data:
            info = self.grid_size(item)
            #h*w used by the sampler for batching
            self.grid_sizes.append(info[0] * info[1])
            #grid dimensions used by the rasterizer
            self.grid_info.append(info)
    
    def __len__(self):
        return self.data.shape[0]
    
    #get transformed data and labels
    def __getitem__(self, id):
        #idx = index of list of data (a 2d ndarr)
        item = self.data[id]

        #Turn training data into a grid
        data_grid, county_mappings, metadata = self.rasterize_data(item, id)

        if self.training:
            #Make matching training label grid
            h, w = metadata["grid_shape"]
            labels_grid = np.zeros((3, h, w), dtype=np.float32)

            for i, mapping in enumerate(county_mappings):
                x = mapping["grid_x"]
                y = mapping["grid_y"]

                labels_grid[0, y, x] = item[i][5]
                labels_grid[1, y, x] = item[i][6]
                labels_grid[2, y, x] = 100 - (item[i][6] + item[i][5])

            target_tensor = torch.from_numpy(labels_grid)

        input_tensor = torch.from_numpy(data_grid)


        if self.training:
            return input_tensor, target_tensor, county_mappings
        else:
            return input_tensor, county_mappings

    def grid_size(self, rawdata, buffer=1.05):
        #Use density of counties to determine grid size
        #print(f"grid_size {type(rawdata)}")
        coords = rawdata[:, [1,2]]
        nn_tree = KDTree(coords, leafsize=50)
        distances, _ = nn_tree.query(coords, k=2)
        min_distance = np.min(distances[:, 1])
        pixel_size = min_distance * 0.5

        lon_min = rawdata[:, 1].min()
        lon_max = rawdata[:, 1].max()
        lat_min = rawdata[:, 2].min()
        lat_max = rawdata[:, 2].max()

        grid_width = int(np.ceil((lon_max - lon_min) / pixel_size * buffer))
        grid_height = int(np.ceil((lat_max - lat_min) / pixel_size * buffer))

        #Reduce grid size if too large for memory
        GRID_MAX = 2048
        if grid_width > GRID_MAX or grid_height > GRID_MAX:
            print(f"Grid size {grid_width} x {grid_height} being reduced for safety: {GRID_MAX} x {GRID_MAX}. May cause collisions")
            scale = GRID_MAX / max(grid_width, grid_height)
            grid_width = int(grid_width * scale)
            grid_height = int(grid_height * scale)
            #pixel_size = min_distance * 0.5

        return [grid_width, grid_height, lon_min, lon_max, lat_min, lat_max, pixel_size]

    #Use latitude and longitude to convert list of points to a 2D grid
    def rasterize_data(self, rawdata, id):
        #print("Converting counties to grid...")

        grid_width, grid_height, lon_min, lon_max, lat_min, lat_max, pixel_size = self.grid_info[id]

        county_mappings = []

        #Channel 0: population. Channel 1: presence/absence of a county
        grid = np.zeros((2, grid_height, grid_width), dtype=np.float32)

        for row in rawdata:
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
                'c_id': rawdata[0],
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

#---------------------------------------------------------------------------------------------------------------
#Custom batch sampler that groups maps into batches and pads them to the same size for better training results
class SizeBasedBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, grid_sizes):
        super().__init__(sampler, batch_size, drop_last)
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

#---------------------------------------------------------------------------------------------------------------

#pad grids in given batch to the largest size and update mappings for batch index
def pad_training(batch):
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
                'id': m['c_id'],
                'batch_id': batch_id,
                'x': m['grid_x'],
                'y': m['grid_y']
            })

    return torch.stack(padded_grids), torch.stack(padded_labels), padded_mappings
    
#ideally make training and testing collation the same function with a bool switch if i have time
#pad grids in given batch to the largest size and update mappings for batch index
def pad_testing(batch):
    data_grids = [item[0] for item in batch]
    mappings = [item[1] for item in batch]

    #pad grids to the largest size grid in this batch
    max_h = max(g.shape[1] for g in data_grids)
    max_w = max(g.shape[2] for g in data_grids)

    padded_grids = []
    padded_mappings = []

    #pad the right and bottom with 0s
    for batch_id, (grid, label, mapping) in enumerate(zip(data_grids, mappings)):
        pad_h = max_h - grid.shape[1]
        pad_w = max_w - grid.shape[2]

        padded_grids.append(torch.nn.functional.pad(grid, (0, pad_w, 0, pad_h), value=0))

        for m in mapping:
            padded_mappings.append({
                'id': m['c_id'],
                'batch_id': batch_id,
                'x': m['grid_x'],
                'y': m['grid_y']
            })

    return torch.stack(padded_grids), padded_mappings
    

#---------------------------------------------------------------------------------------------------------------

def load_data(training=False, sampling=False):
    file_data = []
    folder = input("Enter the location of the folder containing data file(s).\n")
    data_location = os.path.join(folder, "*.csv")
    data_files = glob.glob(data_location)

    #print(f"Found training files: {training_files}")
    if not data_files:
        print("No data files found.")
        return
    else:
        print(f"Found {len(data_files)} data files.\n")

    for file in data_files:
        data = pd.read_csv(file, nrows=100, header=None).to_numpy()
        file_data.append(data)

    #Test the model on 20% of the data bootstrap sampled
    if sampling:
        sample_size = round(len(data) * .2)
        bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
        #todo: finish this

    print("Processing data...")

    file_data = np.array(file_data)

    if training:
        normalize_training_list_np(file_data, 3, normalizer)
        dataset = CountyDataset(file_data, training)
        return dataset
    else:
        normalize_testing(file_data, 3)
        dataset = CountyDataset(file_data, training)
        return dataset, len(data_files)

    #print(f"{normalizer.mean_}, {normalizer.scale_}")
    

#Load training data and train a new model.
def train_model(model):
    dataset = load_data(True)

    base_sampler = SequentialSampler(dataset)
    custom_batch_sampler = SizeBasedBatchSampler(
        base_sampler,
        batch_size=3,
        drop_last=False,
        grid_sizes=dataset.grid_sizes
    )

    dataloader = DataLoader(
        dataset, batch_sampler = custom_batch_sampler,
        collate_fn = pad_training
    )

    print("Starting training...")

    epochs = 7
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    print("\n")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
            
        for i, (padded_grids, padded_labels, mappings) in enumerate(dataloader):
            print(f"Batch tensor shape: {padded_grids.shape}, {padded_labels.shape}")

            coords_list = [[m['batch_id'], m['y'], m['x']] for m in mappings]

            indices = tf.constant(coords_list, dtype=tf.int32)

            cce_loss_fn = tf.keras.losses.CategoricalCrossentropy()

            #gradienttape for backpropagation
            with tf.GradientTape() as tape:
                #batch_size, 3, h, w
                output_maps = model(padded_grids, training=True)
                #move channels to end
                output_maps_permuted = tf.transpose(output_maps, perm=[0, 2, 3, 1])
                #print(type(output_maps))

                #predictions = output_maps[batch_idx, :, y_idx, x_idx]
                #[counties, 3]
                predictions = tf.gather_nd(params=output_maps_permuted, indices=indices)
                #print(predictions.shape)

                pct_predictions = keras.activations.softmax(predictions, axis=1)
                #print(pct_predictions.shape)

                #true_labels = padded_labels[batch_id, :, y_id, x_id]
                padded_labels_transposed = tf.transpose(padded_labels, perm=[0, 2, 3, 1])
                true_labels = tf.gather_nd(params=padded_labels_transposed, indices=indices)
                #true_labels = tf.transpose(true_labels, perm=[1,0])
                #print(true_labels.shape)

                loss = cce_loss_fn(true_labels, pct_predictions)

            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            epoch_loss += float(loss)
            batches += 1

        print(f"Epoch {epoch+1}/{epochs} - Average loss: {epoch_loss / batches:.4f}")
    print("Training complete. Model will be saved to current directory as: jerry_mandarin.keras")
    
    model.save("jerry_mandarin.keras")

    print("Scaler for this model will be saved to current directory as: jm_scaler.bin")
    print("The scaler and model only need to be loaded from file when not in memory.")

    joblib.dump(normalizer, 'jm_scaler.bin')


normalizer = StandardScaler()
model = None


keep_running = True
print("- Jerry Mandarin -")
while keep_running:
    train = input("\nWould you like to train a new model? [Y/N]\n")

    if train.upper() == 'Y':
        model = create_fcn()
        train_model(model)
    
    testing = input("\nWould you like to run a trained model? [Y/N]\n")
    if testing.upper() == 'Y':
        test_model()

    quit = input("Would you like to quit? [Y/N]\n")
    if quit.upper() == 'Y':
        keep_running = False
