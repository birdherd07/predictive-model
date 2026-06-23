# predictive-model
**AI/Data Science Projects I - Predictive model**

Group members: James, Evan, Madelyn, Rachel

# Contents
- **requirements.txt**: Required libraries for model.py
- **model.py**: Program for creating, training and testing a fully convolutional model on census block data.
- **jerry_mandarin.keras**: A pre-trained model file that can be loaded into model.py for testing
- **jm_scaler.bin**: The pre-calculated scaler for jerry_mandarin.keras that can be loaded into model.py for scaling population data
- ***_training_generated.csv**: 9 state training data files used to train the model in the format: ID, LON, LAT, POP, VOTES, R%, D%
  - States: CA, CO, MO, MT, NY, PA, TN, TX, WI   

# Usage
### **Training a new model:**
  - Run model.py
  - When prompted, enter 'y' when asked if you would like to train a new model. **Note** this will overwrite any existing model and/or scaler in the current directory.
  - When prompted, enter the directory of the folder containing all of the csv training files formatted as above.
  - Wait for training to complete.
  - The trained model will be saved to the current directory as 'jerry_mandarin.keras' and its scaler will be saved as 'jm_scaler.bin'.
  - From here, the trained model can be run immediately if desired.
### **Running a trained model:**
  - Run model.py if not already running
  - When prompted, answer 'n' when asked if you would like to train a new model.
  - When prompted, answer 'y' when asked if you would like to run a trained model.
  - Load a trained model:
    - If training was just completed, the trained model and scaler will be used.
      - Otherwise, if there is a trained *.keras model file, in the current directory, it will be found. Answer 'y' when prompted to use it.
      - Otherwise, enter 'n' and then enter the full path of a model in another directory.
  - Load a scaler: As with the model, the calculated scaler will be used if training was just completed.
    - Otherwise, repeat the model load process with a *.bin scaler.
  - When prompted, answer 'n' when asked if you would like to use a bootstrap sample of testing data.
    - Otherwise, answer 'y' to use only a 20% bootstrap sample of the test data.
  - When prompted, enter the directory of the folder containing all of the csv testing files formatted as ID, LON, LAT, POP.
  - When prompted, enter a name for the output csv file of model predictions.
  - Wait for testing to complete.
  - The specified name csv file will be saved to the current directory in the format ID, R%, D%.

# Details
### Data transformation
The input csv files are read, and then data is pre-aggregated by rounding lat and lon. Blocks that have the same rounded lat and lon are mapped to the same point of data to prevent collisions and data loss when they are mapped to a grid. A ledger is kept of IDs if testing, in order to extract them later for outputting to file. The population attribute is normalized by the scaler, the percentages are turned to decimals for better model performance, and then the data is placed in a dataset which rasterizes each list into a 3D list of 2 2D grids. If training, a 3rd label is created of 1 - (R% + D%) for each point. Blocks are mapped to points in the grid by their lat and long. At that point in the grid, one 2D grid holds the population, and the other holds a 1 to mark the presence of a block. The large grids are then sliced into smaller 256x256 grids for performance and training. 

### FCN Model
The model is a fully convolutional model able to take in multiple sized inputs. It outputs 3 classes per data point: R%, D% and remainer% since R% and D% do not always add up to 100, but cannot exceed it. The output is run through a softmax function to ensure this and the remainder output channel is ignored when writing results to file.
