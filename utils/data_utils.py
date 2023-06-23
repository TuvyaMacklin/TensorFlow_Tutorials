import os, datetime

import tensorflow as tf

root_model_dir = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/models"
root_tensorboard_dir = ".tensorboard"

def save_model(model, model_name, root_model_dir = root_model_dir):
    '''
    Saves a model to the given directory.

    Parameters:
        model (tf.keras.Model): The model to save.
        model_name (str): The name of the model.
        root_model_dir (str): The root directory for the models.
    
    Raises:
        FileNotFoundError: If the model directory doesn't exist.
    
    Returns:
        None
    
    Notes:
        The model is saved to root_model_dir/model_name/model.keras.
    '''
    # Ensure that a directory exists for the model
    # The path should be model_dir/model_name
    ensure_directory_exists(os.path.join(root_model_dir, model_name))

    # Save the model
    model.save(os.path.join(root_model_dir, model_name, "model.keras"))

def load_model(model_name, root_model_dir = root_model_dir, path_to_model = None):
    '''
    Loads a model from the given directory.

    Parameters:
        model_name (str): The name of the model.
        root_model_dir (str): The root directory for the models.
        path_to_model (str): The path to the model file. If None, the path is root_model_dir/model_name/model.keras.
    
    Returns:
        The loaded model.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    
    Notes:
        The model is loaded from root_model_dir/model_name/model.keras.
    '''
    # Ensure that the model file exists
    path = None
    if path_to_model is None:
        path = os.path.join(root_model_dir, model_name, "model.keras")
    else:
        path = path_to_model
   
    if not os.path.isfile(path):
        raise FileNotFoundError

    # Load the model
    model = tf.keras.models.load_model(path)
    return model

# Log model results to a csv file with the hyperparameters
def log_results(history, hps, model_name, root_model_dir = root_model_dir):
    '''
    Logs the results of a model to a csv file with the hyperparameters.

    Parameters:
        history (tf.keras.callbacks.History): The history of the model.
        hps (dict): The hyperparameters of the model.
        model_name (str): The name of the model.
        root_model_dir (str): The root directory for the models.

    Returns:
        None
    
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    
    Notes:
        The results are logged to root_model_dir/model_name/logs.csv.
    '''
    # Get the final training and validation accuracy
    accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    # Ensure that a directory exists for the model results
    # The path should be model_dir/model_name/logs.csv
    ensure_directory_exists(os.path.join(root_model_dir, model_name))
    csv_path = os.path.join(root_model_dir, model_name, "logs.csv")

    # Create a csv file if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("date and time,accuracy,val_accuracy,img_height,img_width,batch_size,epochs,learning_rate\n")

    # Append the results to the csv file
    with open(csv_path, "a") as f:
        f.write(f"{datetime.datetime.now()},{accuracy},{val_accuracy},{hps['img_height']},{hps['img_width']},{hps['batch_size']},{hps['epochs']},{hps['learning_rate']}\n")

def ensure_directory_exists(path):
    '''
    Ensures that a directory exists at the given path. If it doesn't exist, it creates it.

    Parameters:
        path (str): The path to the directory.
    
    Notes:
        If the directory already exists, nothing happens.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def get_tensorboard_callback(model_name: str):
    '''
    Returns a TensorBoard callback for a model with the given name.

    Parameters:
        model_name (str): The name of the model.
    
    Returns:
        A TensorBoard callback.
    
    Notes:
        The TensorBoard callback is saved to root_tensorboard_dir/fit/model_name/current_date_and_time.

        To view the TensorBoard, run the following command in the terminal:
        ```
        tensorboard --logdir root_tensorboard_dir/fit/model_name
        ```
        `root_tensorboard_dir` is the root directory for the TensorBoard callbacks. The default is .tensorboard.
    '''
    log_dir = os.path.join(root_tensorboard_dir, "fit", model_name, datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback