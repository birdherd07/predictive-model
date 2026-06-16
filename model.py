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

#Standardize population column of training data using z-score normalization (collection of state 2d ndarrays)
def normalize_training_list(trainingListData, populationCol: int, normalizer: StandardScaler):
    #Calculate the mean and standard deviation over all training ndarrays
    for trainingData in trainingListData:
        normalizer.partial_fit(trainingData[:, [populationCol]])

    #Transform each frame using the aggregated values
    for trainingData in trainingListData:
        trainingData[:, [populationCol]] = normalizer.transform(trainingData[:, [populationCol]])  
        #turn percentages into decimals for better training
        trainingData[:, 5] *= .01
        trainingData[:, 6] *= .01
    #print(f"{normalizer.mean_}, {normalizer.scale_}")

#Use the normalizer from training to scale the population column of test data.
def normalize_testing_list(testListData, populationCol: int, normalizer: StandardScaler):
    for testData in testListData:
        testData[:, [populationCol]] = normalizer.transform(testData[:, [populationCol]])

#Load test data and use a trained model to make predictions.
def test_model(model):
    global normalizer
    if not model:
        model_file = glob.glob("*.keras")
        if model_file:
            use = input(f"Model found in current directory: {model_file[0]}. Use this model? [Y/N]\n")
            if use.upper() =='Y':
                try:
                    model = models.load_model(model_file[0])
                except:
                    print(f"Model could not be loaded.")
                    return
        else:
            model_loc = input("Enter the full path of the model file.\n")            
            try:
                model = models.load_model(model_loc)
            except:
                print(f"No model found at a {model_loc}.")
                return
        
    try:
        check_is_fitted(normalizer)
    except:
        scaler_file = glob.glob("*.bin")
        if scaler_file:
            use = input(f"Potential scaler file found in current directory: {scaler_file[0]}. Use this scaler? [Y/N]\n")
            if use.upper() == 'Y':
                try:
                    normalizer = joblib.load(scaler_file[0])
                except:
                    print("No scaler file found in current directory.")
                    return
        else:
            scaler_name = input("Enter the full path of the scaler file.\n")
            try:
                normalizer = joblib.load(scaler_name)
            except:
                print("No scaler file found in directory.")
                return
        
    bootstrap = input("Use only a bootstrap sample of data instead of all? [Y/N]\n")
    if bootstrap.upper() == 'Y':
        dataset, ledger = load_data(False, True)
    else:
        dataset, ledger = load_data(False, False)

    length = len(dataset)

    name = input("Enter a name for the predictions file.\n ")

    print(f"Predictions will be written to file as: {name}.csv\n")

    #predictions = {}
    predictions = []

    processed_ids = set()
    #run model
    for i in range(length):
        print(f"Predicting batch {i+1} of {length}")
        #3, h, w
        input_tensor, county_mappings = dataset[i]
        model_input = tf.expand_dims(input_tensor, axis=0)

        output_map = model(model_input, training=False)
        pct_map = tf.keras.activations.softmax(output_map, axis=1)

        #3, h, w
        pct_map = tf.squeeze(pct_map, axis=0)
        
        for mapping in county_mappings:
            #id = mapping['c_id']
            spatial_key = mapping["c_id"]
            x = mapping['grid_x']
            y = mapping['grid_y']

            r_pct = float(pct_map[0, y, x]) * 100
            d_pct = float(pct_map[1, y, x]) * 100

            if spatial_key in ledger:
                for array_id, row_id, original_id in ledger[spatial_key]:
                    if original_id not in processed_ids:
                        processed_ids.add(original_id)
                        predictions.append([original_id, r_pct, d_pct])

    #         predictions[id] = {
    #             "R": r_pct,
    #             "D": d_pct
    #         }

    # final_predictions = [
    #     {"county_id": cid, **data} for cid, data in predictions.items()
    # ]

    print(len(predictions))


    #Output list of ids and predictions to csv
    #format: id, R%, D%
    with open(f'{name}.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)
        # for row in final_predictions:
        #     writer.writerow(row.values())
        # for item in results:
        #     writer.writerow([key, value])

    print("Predictions complete.\n")



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
    
    # convolutional block 2: dilation to wider area. idk man this seems to make it worse
    # x = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=2, data_format='channels_first')(x)
    # #x = layers.BatchNormalization(axis=1)(x)
    # x = layers.Activation('relu')(x)

    # convolutional block 3
    x = layers.Conv2D(64, kernel_size=5, padding='same', data_format='channels_first')(x)
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
    def __init__(self, data_list, training: bool, patch_size=256, stride=192):
        #data_list: a 3d ndarr.
        #each 2d ndarr is a group of counties of form of format 
        #ID LON LAT POP VOTES R% D% (training = True)
        #or 
        #ID LON LAT POP (training = False) omits labels
        self.training = training

        self.patch_size = patch_size
        self.stride = stride
        self.input_patches = []
        self.target_patches = []
        self.patch_mappings = []

        for item in data_list:
            large_input, large_target, large_mappings = self.rasterize_data(item)

            self._slice_and_store(large_input, large_target, large_mappings)
    
    def __len__(self):
        #return self.data.shape[0]
        return len(self.input_patches)

    #get transformed data and labels
    def __getitem__(self, id):
        input_tensor = self.input_patches[id]
        mapping = self.patch_mappings[id]
        
        if self.training:
            target_tensor = self.target_patches[id]
            return input_tensor, target_tensor, mapping
        else:
            return input_tensor, mapping
        
    def _slice_and_store(self, large_input, large_target, large_mappings):
        #divide big grids into smaller grids for performance
        channels, H, W = large_input.shape
        #target_channels, _, _ = large_target.shape
        
        # remap grid using stride sliding window
        for y_start in range(0, H - self.patch_size + 1, self.stride):
            for x_start in range(0, W - self.patch_size + 1, self.stride):
                
                #Check if this patch window actually contains any counties
                local_mappings = []
                for county in large_mappings:
                    #where the county falls relative to this patch's top-left corner
                    local_y = county["grid_y"] - y_start
                    local_x = county["grid_x"] - x_start
                    
                    #If the coordinate sits comfortably inside the 256x256 window, save it
                    if 0 <= local_y < self.patch_size and 0 <= local_x < self.patch_size:
                        #Copy the county dict and update its coordinates to the local patch space
                        updated_county = county.copy()
                        updated_county["grid_y"] = local_y
                        updated_county["grid_x"] = local_x
                        local_mappings.append(updated_county)
                
                #If this patch has at least one county in it, save it as a valid training sample
                if len(local_mappings) > 0:
                    #Slice out the physical 256x256 tensor blocks
                    input_patch = large_input[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                    if self.training:
                        target_patch = large_target[:, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]

                    #Append to dataset
                    self.input_patches.append(input_patch)
                    if self.training:
                        self.target_patches.append(target_patch)
                    self.patch_mappings.append(local_mappings)

    #Use latitude and longitude to convert list of points to a 2D grid
    def rasterize_data(self, rawdata, buffer=1.05):
        #print("Converting counties to grid...")
        #Use density of counties to determine grid size
        #print(f"grid_size {type(rawdata)}")
        coords = rawdata[:, [1,2]]
        nn_tree = KDTree(coords, leafsize=50)
        distances, _ = nn_tree.query(coords, k=2)
        min_distances = distances[:, 1][distances[:, 1] > 0]

        if len(min_distances) > 0:
            min_distance = np.min(min_distances)
        else:
            min_distance = 0.01
        pixel_size = min_distance * 0.5

        lon_min = rawdata[:, 1].min()
        lon_max = rawdata[:, 1].max()
        lat_min = rawdata[:, 2].min()
        lat_max = rawdata[:, 2].max()

        grid_width = int(np.ceil((lon_max - lon_min) / pixel_size * buffer))
        grid_height = int(np.ceil((lat_max - lat_min) / pixel_size * buffer))

        #Reduce grid size if too large for memory
        GRID_MAX = 8192
        if grid_width > GRID_MAX or grid_height > GRID_MAX:
            #print(f"Grid size {grid_width} x {grid_height} being reduced for safety: {GRID_MAX} x {GRID_MAX}. May cause collisions")
            scale = GRID_MAX / max(grid_width, grid_height)
            grid_width = int(grid_width * scale)
            grid_height = int(grid_height * scale)
            pixel_size = min_distance * 0.5

        county_mappings = []

        #Channel 0: population. Channel 1: presence/absence of a county
        grid = np.zeros((2, grid_height, grid_width), dtype=np.float32)

        occupied_pixels = set()
        clash_count = 0

        for row in rawdata:
            # Map to [0, width-1] and [0, height-1]
            x = int(np.round(((row[1] - lon_min) / (lon_max - lon_min)) * (grid_width - 1)))
            y = int(np.round(((row[2] - lat_min) / (lat_max - lat_min)) * (grid_height - 1)))

            pxcoord = (y, x)
            if pxcoord in occupied_pixels:
                clash_count += 1
            else:
                occupied_pixels.add(pxcoord)

            #Make north at top of grid
            invert_y = (grid_height - 1) - y

            #Place population, presence at point in grid
            grid[0, invert_y, x] += row[3]
            grid[1, invert_y, x] = 1.0


            spatial_key = f"{row[1]}_{row[2]}"

            #For retrieving counties by ID from FCN output
            county_mappings.append({
                #'c_id': row[0],
                'c_id': spatial_key,
                'grid_x': x,
                'grid_y': invert_y
            })

        if self.training:
            #Make matching training label grid
            h, w = grid_height, grid_width
            labels_grid = np.zeros((3, h, w), dtype=np.float32)

            for i, mapping in enumerate(county_mappings):
                x = mapping["grid_x"]
                y = mapping["grid_y"]

                labels_grid[0, y, x] = rawdata[i][5]
                labels_grid[1, y, x] = rawdata[i][6]
                labels_grid[2, y, x] = 1.0 - (rawdata[i][6] + rawdata[i][5])

            target_tensor = torch.from_numpy(labels_grid)

        input_tensor = torch.from_numpy(grid)

        print(f"Clash count: {clash_count}")

        if self.training:
            return input_tensor, target_tensor, county_mappings
        else:
            return input_tensor, None, county_mappings


def patch_collate_fn(batch):
    """
    batch is a list of tuples coming from __getitem__:
    [(input_tensor, target_tensor, mapping), (input_tensor, target_tensor, mapping), ...]
    """
    # 1. Zip the components out of the batch structure
    inputs, targets, mappings = zip(*batch)
    
    # 2. Natively stack the identical 256x256 tensors 
    # Resulting shapes: [Batch_Size, Channels, 256, 256]
    batched_inputs = torch.stack(inputs, dim=0)
    batched_targets = torch.stack(targets, dim=0)
    
    # 3. Flatten out the county mappings list and append the correct batch index (b_idx)
    flattened_mappings = []
    for batch_index, county_list in enumerate(mappings):
        for county in county_list:
            updated_county = county.copy()
            updated_county['batch_id'] = batch_index  # Track which element in the batch this county belongs to
            flattened_mappings.append(updated_county)
            
    return batched_inputs, batched_targets, flattened_mappings

#---------------------------------------------------------------------------------------------------------------
def pre_aggregate_data(list_of_arrays, training=False, precision=2):
    #Each row in an array is: [id, lat, lon, population, percent_a, percent_b]
    
    #Returns a list of clean 2D numpy arrays where colliding points are 
    #aggregated, but keeps a reference ledger to unpack later.
    # Global dictionary to map rounded spatial coordinates to their ledger
    coordinate_buckets = {}

    for array_idx, arr in enumerate(list_of_arrays):
        for row_idx in range(arr.shape[0]):
            row = arr[row_idx]
            
            entity_id = row[0]
            lat       = row[1]
            lon       = row[2]
            pop       = row[3]
            if training:
                votes     = row[4]
                pct_a     = row[5]
                pct_b     = row[6]
            
            # Create a unique string key by rounding coordinates
            rounded_lat = round(lat, precision)
            rounded_lon = round(lon, precision)
            spatial_key = f"{rounded_lat}_{rounded_lon}"
            
            if spatial_key not in coordinate_buckets:
                coordinate_buckets[spatial_key] = {
                    "lat": rounded_lat,
                    "lon": rounded_lon,
                    "total_pop": 0.0,
                    "sum_weighted_a": 0.0,
                    "sum_weighted_b": 0.0,
                    "original_rows": []  # ledger of original rows
                }
            
            # Handle population-weighted targets
            weight = max(pop, 1.0) # Avoid division by zero if pop is 0
            coordinate_buckets[spatial_key]["total_pop"] += weight
            if training:
                coordinate_buckets[spatial_key]["sum_weighted_a"] += (pct_a * weight)
                coordinate_buckets[spatial_key]["sum_weighted_b"] += (pct_b * weight)
            
            # Track exactly where this row came from to unpack it later
            # Storing (array_idx, row_idx, original_id)
            coordinate_buckets[spatial_key]["original_rows"].append((array_idx, row_idx, entity_id))

    # build one aggregated array per state/file matching original list length
    aggregated_lists = [[] for _ in range(len(list_of_arrays))]
    
    # Global tracking ledger to pass to test loop later
    # Format: { spatial_node_key: [(array_idx, row_idx, entity_id), ...] }
    extraction_ledger = {}

    for key, bucket in coordinate_buckets.items():
        total_p = bucket["total_pop"]
        
        if training:
            # Calculate the proper weighted average percentages for this single pixel
            mean_a = bucket["sum_weighted_a"] / total_p
            mean_b = bucket["sum_weighted_b"] / total_p
            
        # assign this aggregated pixel node to the array_idx of its FIRST original point
        # to preserve state-by-state file grouping.
        primary_array_idx = bucket["original_rows"][0][0]

        if training:       
            # Synthesize the new compressed row
            # use a dummy ID (-999) or the bucket hash string converted to float
            aggregated_row = [
                -999.0,          # Dummy ID for the model layer
                bucket["lat"],   
                bucket["lon"],   
                total_p,
                votes,         
                mean_a,          
                mean_b           
            ]
        else:
            aggregated_row = [
                -999.0,          
                bucket["lat"],   
                bucket["lon"],   
                total_p         
            ]
        
        aggregated_lists[primary_array_idx].append(aggregated_row)
        
        if not training:
            # Save the map coordinates to the extraction ledger using the key
            extraction_ledger[key] = bucket["original_rows"]

    # Convert the sub-lists back into 2D NumPy arrays
    output_list_of_arrays = []
    for sub_list in aggregated_lists:
        if len(sub_list) > 0:
            output_list_of_arrays.append(np.array(sub_list))
        else:
            output_list_of_arrays.append(np.empty((0, 6))) # Keep empty placeholders intact
    if training:
        return output_list_of_arrays, None
    else:
        return output_list_of_arrays, extraction_ledger


def load_data(training=False, sampling=False):
    file_data = []
    folder = input("Enter the location of the folder containing data file(s).\n")
    data_location = os.path.join(folder, "*.csv")
    data_files = glob.glob(data_location)

    #print(f"Found training files: {training_files}")
    if not data_files:
        print("No csv data files found.")
        return
    else:
        print(f"Found {len(data_files)} csv data files.\n")

    for file in data_files:
        data = pd.read_csv(file, dtype={0:str}, header=None).to_numpy()

        #Test the model on 20% of the data
        if not training and sampling:
            sample_size = round(len(data) * .2)
            rng = np.random.default_rng()
            row_indices = rng.choice(data.shape[0], size=sample_size, replace=True)
            bootstrap_sample = data[row_indices, :]
            #bootstrap_sample = np.random.choice(data, size=sample_size, replace=True)
            file_data.append(bootstrap_sample)
        else:
            file_data.append(data)

    print("Processing data...")
    #TODO training switch
    file_data, ledger = pre_aggregate_data(file_data, training)

    if training:
        normalize_training_list(file_data, 3, normalizer)
    else:
        normalize_testing_list(file_data, 3, normalizer)
        
    dataset = CountyDataset(file_data, training)

    return dataset, ledger

    #print(f"{normalizer.mean_}, {normalizer.scale_}")
    

#Load training data and train a new model.
def train_model(model):
    global normalizer
    dataset, _ = load_data(True)

    BATCH_SIZE = 32

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=patch_collate_fn,
        drop_last=False
    )

    print("Starting training...")

    epochs = 7
    optimizer = keras.optimizers.Adam(learning_rate=.0001)

    print("\n")
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
            
        for i, (padded_grids, padded_labels, mappings) in enumerate(dataloader):
            #print(f"Batch tensor shape: {padded_grids.shape}, {padded_labels.shape}")
            print(f"Batch {i+1}")

            coords_list = [[m['batch_id'], m['grid_y'], m['grid_x']] for m in mappings]

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

        print(f"Epoch {epoch+1}/{epochs} - Average loss: {epoch_loss / batches:.4f}\n")
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
        test_model(model)

    quit = input("Would you like to quit? [Y/N]\n")
    if quit.upper() == 'Y':
        keep_running = False
