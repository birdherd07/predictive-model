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
  - Wait for testing to complete. The specified name csv file will be savedto the current directory in the format ID, R%, D%.

