import os

root_model_dir = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/models"

def save_model(model, model_name, root_model_dir = root_model_dir):
    # Ensure that a directory exists for the model
    # The path should be model_dir/model_name
    ensure_directory_exists(os.path.join(root_model_dir, model_name))

    # Save the model
    model.save(os.path.join(root_model_dir, model_name, "model.keras"))

# Log model results to a csv file with the hyperparameters
def log_results(history, hps, model_name, root_model_dir = root_model_dir):
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
    if not os.path.exists(path):
        os.mkdir(path)