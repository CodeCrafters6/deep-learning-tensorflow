# helpers
import os
import random
from pathlib import Path
import shutil
from shutil import copyfile
import datetime
import pytz
import pickle

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import Callback, EarlyStopping,  ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm.notebook import tqdm
from itertools import cycle
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc


def delete_dir(dir_name):
    shutil.rmtree(f'/content/{dir_name}') #deletes a directory and all its contents.

def split_dataset_train_test(
    src_directory, dest_directory, split_ratio=0.2, random_state=42, only_train_n_test=True
):
    """
    Split a dataset into train, test, and validation sets and copy the files to the respective directories.

    Args:
        src_directory (str): Path to the directory containing the image dataset.
        dest_directory (str): Path to the directory where the train, test, and validation sets will be copied.
        split_ratio (float): The proportion of the dataset to include in the test and validation sets (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).
        only_train_n_test (bool): Whether to generate only train and test sets or also include a validation set (default: True).
    """

    # Create train and test directories
    train_dir = Path(dest_directory) / "train"
    test_dir = Path(dest_directory) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if not only_train_n_test:
        # Create validation directory
        valid_dir = Path(dest_directory) / "valid"
        valid_dir.mkdir(parents=True, exist_ok=True)

    # Get all the image files in the source directory and their corresponding labels
    root = Path(src_directory)
    files = list(root.glob("*/*"))
    labels = [str(file.parent).split('/')[-1] for file in files]

    # Split the files and labels into train, test/validation sets
    train_files, test_valid_files, train_labels, test_valid_labels = train_test_split(
        files, labels, test_size=split_ratio, random_state=random_state, stratify=labels
    )

    if not only_train_n_test:
        # Split the test_valid_files and labels into test and validation sets
        test_files, valid_files, test_labels, valid_labels = train_test_split(
            test_valid_files, test_valid_labels, test_size=0.5, random_state=random_state, stratify=test_valid_labels
        )

        # Copy validation files to validation directory
        for file in tqdm(valid_files, desc="Copying validation files"):
            dest_path = str(file).replace(src_directory, str(valid_dir))
            Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file, dest_path)

        # Print the number of files in the validation set
        print("Number of validation files:", len(valid_files))

    else:
        test_files = test_valid_files
        test_labels = test_valid_labels

    # Copy train files to train directory
    for file in tqdm(train_files, desc="Copying train files"):
        dest_path = str(file).replace(src_directory, str(train_dir))
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_path)

    # Copy test files to test directory
    for file in tqdm(test_files, desc="Copying test files"):
        dest_path = str(file).replace(src_directory, str(test_dir))
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest_path)

    # Print the number of files in the train and test sets
    print("Number of train files:", len(train_files))
    print("Number of test files:", len(test_files))


def create_subset_dataset(dataset_path, percentage, new_path):
    """
    Creates a subset dataset by randomly selecting a percentage of images from the "train" subdirectory
    in the original dataset.

    Args:
        dataset_path (str): The path to the original dataset directory.
        percentage (float): The percentage of images to include in the subset (between 0 and 100).
        new_path (str): The path to the directory where the subset dataset will be created.

    Returns:
        None

    """
    print(f"Creating {percentage}% data subset for the 'train' subdirectory...")
    subset_percentage = percentage / 100

    # Set the paths for the original and new directories
    dataset_path = Path(dataset_path)
    new_directory = Path(new_path)

    # Keep the "test" and "val" directories unchanged
    tvt_dirs = [
        subdir
        for subdir in dataset_path.iterdir()
        if subdir.is_dir() and subdir.name != "train"
    ]

    for tvt in tvt_dirs:
        tvt_path = new_directory / tvt.name
        tvt_path.mkdir(parents=True, exist_ok=True)

        # Copy the subdirectories as they are
        for class_path in tvt.iterdir():
            class_name = class_path.name
            new_class_path = tvt_path / class_name
            new_class_path.mkdir(parents=True, exist_ok=True)
            for image_path in class_path.iterdir():
                new_image_path = new_class_path / image_path.name
                copyfile(image_path, new_image_path)

    # Create a subset for the "train" subdirectory
    train_path = dataset_path / "train"
    train_new_path = new_directory / "train"
    train_new_path.mkdir(parents=True, exist_ok=True)

    class_stats = {}
    for class_path in train_path.iterdir():
        class_name = class_path.name
        class_stats[class_name] = list(class_path.glob("*"))

    # Calculate the size of the new dataset based on the subset percentage
    total_count = sum(len(images) for images in class_stats.values())
    new_dataset_size = int(total_count * subset_percentage)

    new_dataset = []
    print(f"train: {total_count}/{new_dataset_size} images selected.")

    for class_name in class_stats.keys():
        images = class_stats[class_name]
        num_images = len(images)

        num_selected_images = int(new_dataset_size / len(class_stats))
        # Randomly select the images from the current class
        selected_images = random.sample(images, num_selected_images)
        # Add the selected images to the new dataset
        new_dataset.extend(selected_images)

        # Copy the selected images to the new directory
        for image_path in new_dataset:
            class_name = image_path.parent.name
            new_image_path = train_new_path / class_name / image_path.name
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(image_path, new_image_path)

    # Calculate and print statistics
    print("Selected images per class:")
    for class_name, images in class_stats.items():
        selected_images = [image for image in images if image in new_dataset]
        selected_count = len(selected_images)
        total_count = len(images)
        print(
            f"  > Class: {class_name}, Selected: {selected_count}/{total_count} images"
        )

    print("Subset dataset created successfully!\n")


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


# View an image
def view_random_image(root_path, file_extension):
    """
    root_path: root folder path
    file_extension: extension of the image files to search for (e.g., "jpg", "png")
    """
    # Get all image file paths with the specified extension in the directory
    image_paths = list(Path(root_path).rglob(f"*.{file_extension}"))

    if len(image_paths) == 0:
        print("No image files found with the specified extension in the specified directory.")
        return

    # Select a random image path
    random_image_path = random.choice(image_paths)
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.title(str(random_image_path.parent.name))
    plt.axis("off")

    # Get the image size
    height, width, channels = img.shape

    # Print the size statistics
    print("Image Path:", random_image_path)
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)



# Plot loss curves of a model with matplotlib with smothing effect
def smooth_curve(points, factor=0.8):
    """Smooths a curve using exponential moving averages.

    Args:
        points (list): List of values representing the curve.
        factor (float): Smoothing factor (default: 0.8).

    Returns:
        numpy.ndarray: Smoothed curve.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

def plot_loss_curves_mplt(history, 
                          smoothing_factor=0.8, 
                          fill_a=0,
                          with_best_point=False,
                          plt_style="seaborn-v0_8-whitegrid",
                          start_epoch=1, 
                          figsize = (20, 8)
                          ):
    """Plots training curves of a results dictionary with smoothed curves and saves them as PDF.

    Args:
        history (dict): Dictionary containing training history, e.g.
            {"loss": [...],
             "val_loss": [...],
             "accuracy": [...],
             "val_accuracy": [...]}.
        smoothing_factor (float): Smoothing factor for curves (default: 0.7).
        start_epoch (int): Starting epoch number (default: 1).
        fill_a (float): Alpha value for filling the area between curves (default: 0).
        with_best_point (bool): Whether to include the best epoch point on the plot (default: False).
        plt_style (str): Matplotlib style to use for the plot (default: "seaborn-v0_8-whitegrid").
        figsize (tuple): Figure size (width, height) in inches (default: (10, 6)).
    """
    loss = history.history["loss"]
    loss = smooth_curve(loss, smoothing_factor)
    val_loss = history.history["val_loss"]
    val_loss = smooth_curve(val_loss, smoothing_factor)

    accuracy = history.history["accuracy"]
    accuracy = smooth_curve(accuracy, smoothing_factor)
    val_accuracy = history.history["val_accuracy"]
    val_accuracy = smooth_curve(val_accuracy, smoothing_factor)

    epochs = range(start_epoch, len(history.history["loss"])+1)

    index_loss = np.argmin(val_loss)  # This is the epoch with the lowest validation loss
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_accuracy)
    acc_highest = val_accuracy[index_acc]

    sc_label = 'Best epoch = ' + str(index_loss + start_epoch)
    vc_label = 'Best epoch = ' + str(index_acc + start_epoch)

    plt.figure(figsize=figsize, facecolor='white')
    plt.style.use(plt_style)

    # Plot loss
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()  # Get the current axis
    ax1.plot(epochs, loss, '#0570b9', label='Training loss', linewidth=4)
    ax1.plot(epochs, val_loss,  "#ff8000", label='Validation loss', linewidth=4)
    if fill_a:
        ax1.fill_between(epochs, val_loss, loss, color='gray', alpha=fill_a)
    ax1.set_title("Training and Validation Loss", fontsize=20)
    ax1.set_xlabel("Epochs", fontsize=20)
    ax1.set_ylabel("Loss", fontsize=20)
    ax1.set_xlim([0.5, len(epochs)]) 
    # Plot best point
    if with_best_point:
        ax1.scatter(index_loss + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    ax1.legend(fontsize=18, loc='upper right', frameon=False)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()  # Get the current axis
    ax2.plot(epochs, accuracy, '#0570b9',  label='Training Accuracy', linewidth=4)
    ax2.plot(epochs, val_accuracy,"#ff8000", label='Validation Accuracy', linewidth=4)
    if fill_a:
        ax2.fill_between(epochs, val_accuracy, accuracy, color='gray', alpha=fill_a)
    ax2.set_title("Training and Validation Accuracy", fontsize=20)
    ax2.set_xlabel("Epochs", fontsize=20)
    ax2.set_ylabel("Accuracy", fontsize=20)
    ax2.set_xlim([0.5, len(epochs)])
    ax2.set_ylim([0, 1.1])
    # Plot best point
    if with_best_point:
        ax2.scatter(index_acc + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    ax2.legend(fontsize=18, loc='lower right', frameon=False)

    plt.tight_layout()
    plt.show()

def save_pdf_loss_curves_mplt(history, 
                              smoothing_factor=0.8, 
                              fill_a=0,
                              with_best_point=False,
                              plt_style="seaborn-v0_8-whitegrid",
                              start_epoch=1,
                              figsize = (10, 6)
                              ):
    """Save pdf plots of training curves of a results dictionary with smoothed curves and saves them as PDF.

    Args:
        history (dict): Dictionary containing training history, e.g.
            {"loss": [...],
             "val_loss": [...],
             "accuracy": [...],
             "val_accuracy": [...]}.
        smoothing_factor (float): Smoothing factor for curves (default: 0.7).
        start_epoch (int): Starting epoch number (default: 1).
        fill_a (float): Alpha value for filling the area between curves (default: 0).
        with_best_point (bool): Whether to include the best epoch point on the plot (default: False).
        plt_style (str): Matplotlib style to use for the plot (default: "seaborn-v0_8-whitegrid").
        figsize (tuple): Figure size (width, height) in inches (default: (10, 6)).
    """
    loss = history.history["loss"]
    loss = smooth_curve(loss, smoothing_factor)
    val_loss = history.history["val_loss"]
    val_loss = smooth_curve(val_loss, smoothing_factor)

    accuracy = history.history["accuracy"]
    accuracy = smooth_curve(accuracy, smoothing_factor)
    val_accuracy = history.history["val_accuracy"]
    val_accuracy = smooth_curve(val_accuracy, smoothing_factor)

    epochs = range(start_epoch, len(history.history["loss"])+1)

    index_loss = np.argmin(val_loss)  # This is the epoch with the lowest validation loss
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_accuracy)
    acc_highest = val_accuracy[index_acc]

    sc_label = 'Best epoch = ' + str(index_loss + start_epoch)
    vc_label = 'Best epoch = ' + str(index_acc + start_epoch)

    plt.figure(figsize=figsize, facecolor='white')
    plt.style.use(plt_style)

    # Plot loss
    plt.plot(epochs, loss, '#0570b9', label='Training loss', linewidth=4)
    plt.plot(epochs, val_loss, "#ff8000", label='Validation loss', linewidth=4)
    if fill_a:
        plt.fill_between(epochs, val_loss, loss, color='gray', alpha=fill_a)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xlim([0.5, len(epochs)]) 
    # Plot best point
    if with_best_point:
        plt.scatter(index_loss + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    plt.legend(fontsize=10, loc='upper right', frameon=False)

    # Save loss plot separately
    plt.savefig("loss_plot.pdf", bbox_inches='tight')
    plt.clf()  # Clear the figure

    # Plot accuracy
    plt.plot(epochs, accuracy, '#0570b9',  label='Training Accuracy', linewidth=4)
    plt.plot(epochs, val_accuracy,"#ff8000", label='Validation Accuracy', linewidth=4)
    if fill_a:
        plt.fill_between(epochs, val_accuracy, accuracy, color='gray', alpha=fill_a)
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlim([0.5, len(epochs)]) 
    plt.ylim([0, 1.1])
    # Plot best point
    if with_best_point:
        plt.scatter(index_acc + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    plt.legend(fontsize=10, loc='lower right', frameon=False)

    # Save accuracy plot separately
    plt.savefig("accuracy_plot.pdf", bbox_inches='tight')
    plt.clf()  # Clear the figure

    plt.close()

# Plot loss curves of a model using plotly.py with smoothing effect
def plot_loss_curves_plotly(history, smoothing_factor=0.8):
    loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    test_accuracy = history.history["val_accuracy"]
    epochs = list(range(len(loss)))

    smoothed_loss = smooth_curve(loss, smoothing_factor)
    smoothed_test_loss = smooth_curve(test_loss, smoothing_factor)
    smoothed_accuracy = smooth_curve(accuracy, smoothing_factor)
    smoothed_test_accuracy = smooth_curve(test_accuracy, smoothing_factor)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))

    # Plot loss
    fig.add_trace(
        go.Scatter(
            x=epochs, y=smoothed_loss, mode="lines", name="train_loss", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=smoothed_test_loss, mode="lines", name="val_loss", line=dict(color="red")
        ),
        row=1,
        col=1,
    )

    # Plot accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=smoothed_accuracy,
            mode="lines",
            name="train_accuracy",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=smoothed_test_accuracy,
            mode="lines",
            name="val_accuracy",
            line=dict(color="red"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=500, width=1000, title_text="Training Curves")

    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    fig.show()


# Create tensorboard callback (functionized because need to create a new one for each model)


def create_tensorboard_callback(dir_name, experiment_name, timezone="Asia/Dhaka"):
    """
    Creates a TensorBoard callback for logging training metrics during model training.

    Parameters:
    dir_name (str): Directory name where the TensorBoard logs will be saved.
    experiment_name (str): Name of the experiment or model for organizing the logs.
    timezone (str): Timezone for log directory naming. Default is "Asia/Dhaka".

    Returns:
    tf.keras.callbacks.TensorBoard: TensorBoard callback for logging training metrics.
    """
    # Set the time zone based on the provided argument
    target_time_zone = pytz.timezone(timezone)

    # Get the current time in the system's local time zone
    current_time = datetime.datetime.now()

    # Convert the current time to the target time zone
    target_current_time = current_time.astimezone(target_time_zone)

    # Format the current time for the log directory name
    log_time = target_current_time.strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = f"{dir_name}/{experiment_name}/{log_time}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_feature_extractor_model(model_url, IMAGE_SHAPE, num_classes):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

    Args:
      model_url (str): A TensorFlow Hub feature extraction URL.
      num_classes (int): Number of output neurons in output layer,
        should be equal to number of target classes, default 10.

    Returns:
      An uncompiled Keras Sequential model with model_url as feature
      extractor layer and Dense output layer with num_classes outputs.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False,  # freeze the underlying patterns
        name="feature_extraction_layer",
        input_shape=IMAGE_SHAPE + (3,),
    )  # define the input image shape

    # Create our own model
    model = tf.keras.Sequential(
        [
            feature_extractor_layer,  # use the feature extraction layer as the base
            tf.keras.layers.Dense(
                num_classes, activation="softmax", name="output_layer"
            ),  # create our own output layer
        ]
    )

    return model


def augment_random_image(root_path, file_extension, data_augmentation):
    """
    Reads a random image from a randomly chosen class in the target directory,
    applies data augmentation to the image, and plots the original and augmented images side by side.

    Parameters:
    root_path (str): Path to the root directory containing the target images.
    file_extension (str): Extension of the image files.
    data_augmentation (tf.keras.Sequential): Data augmentation model to apply to the images.

    Returns:
    None

    Usage:
    import tensorflow as tf
    from tensorflow.keras import layers

    augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomZoom(0.2),
    ])

    # Define the target directory
    root_path = "data"
    file_extension = "jpg"
    augment_random_image(root_path, file_extension, augmentation) 

    """
    # Get a random class from the target directory
    target_class = random.choice(list(Path(root_path).rglob('*/*')))

    # Get a random image from the chosen class
    class_dir = target_class.parent
    random_image = random.choice(list(class_dir.glob(f'*.{file_extension}')))
    random_image_path = random_image

    # Read the random image
    img = mpimg.imread(random_image_path)

    # Augment the image
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))

    # Plot the original and augmented images
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original random image from class: {target_class.parent.name}")
    plt.axis(False)

    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmented_img))
    plt.title(f"Augmented random image from class: {target_class.parent.name}")
    plt.axis(False)
    plt.show()


def compare_histories_plotly(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects and plots the training and validation accuracy and loss using Plotly.

    Parameters:
    original_history (keras.callbacks.History): History object of the original model.
    new_history (keras.callbacks.History): History object of the new model.
    initial_epochs (int): Number of initial epochs before fine-tuning. Default is 5.

    Returns:
    None
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Training and Validation Accuracy",
            "Training and Validation Loss",
        ),
    )

    # Add traces to the first subplot (Training Accuracy and Validation Accuracy)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_acc))),
            y=total_acc,
            mode="lines",
            name="Training Accuracy",
            marker=dict(color="blue"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_val_acc))),
            y=total_val_acc,
            mode="lines",
            name="Validation Accuracy",
            marker=dict(color="red"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial_epochs - 1, initial_epochs - 1],
            y=[min(total_acc), max(total_acc)],
            mode="lines",
            name="Fine-tune starting point",
            marker=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)

    # Add traces to the second subplot (Training Loss and Validation Loss)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_loss))),
            y=total_loss,
            mode="lines",
            name="Training Loss",
            marker=dict(color="blue"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(total_val_loss))),
            y=total_val_loss,
            mode="lines",
            name="Validation Loss",
            marker=dict(color="red"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[initial_epochs - 1, initial_epochs - 1],
            y=[min(total_loss), max(total_loss)],
            mode="lines",
            name="Fine-tune starting point",
            marker=dict(color="black"),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    # Update layout and display the figure
    fig.update_layout(
        height=1000,
        width=1000,
        showlegend=True,
        title="Comparison of Training and Validation Metrics",
    )
    fig.show()


def compare_histories_mplt(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Validation Accuracy")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Validation Loss")
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1], plt.ylim(), label="Start Fine Tuning"
    )  # reshift plot around epochs
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()



class SaveHistoryCallback(Callback):
    def __init__(self, filepath):
        super(SaveHistoryCallback, self).__init__()
        self.filepath = filepath

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)
        with open(self.filepath, 'wb') as file:
            pickle.dump(self.history, file)
  
def get_callbacks(EarlyStoppingPatience=5,LearningRatePatience=3):
    # Stop training when a monitored metric has stopped improving.
    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=EarlyStoppingPatience, #Number of epochs with no improvement after which training will be stopped.
                               restore_best_weights=True)

    # Reduce learning rate when a metric has stopped improving.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.1,
                                  patience=LearningRatePatience, #Number of epochs with no improvement after which learning rate will be reduced.
                                  verbose=1)

    checkpoint = ModelCheckpoint(filepath='model_weights.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min',
                                 verbose=1)
    # Add other performance-enhancing callbacks if desired
    # For example:
    # tensorboard = TensorBoard(log_dir='logs')
    save_history_callback = SaveHistoryCallback('history.pkl')

    callbacks = [early_stop, checkpoint, reduce_lr, save_history_callback]

    return callbacks

# Load history object
class H:
    def __init__(self,history):
        self.history=  history

def load_history(filepath):
    """
    Example usage: 
    loaded_history = load_history('history.pkl')
    """
    with open(filepath, 'rb') as file:
        history = pickle.load(file)
    return H(history)

import numpy as np

def get_predictions_and_labels(test_data, model):
    """
    Get model predictions and labels from test data.

    Parameters:
        test_data (data generator): The test data generator.
        model (keras.Model): The model for predictions.

    Returns:
        y_prob (array): Predicted probabilities for each class.
        y_pred (array): Predicted class indices.
        y_true (array): True class indices.
        y_true_one_hot (array): One-hot matrix of true labels.

    Example: y_prob, y_pred, y_true, y_true_one_hot = get_predictions_and_labels(test_data, model = combined_model)
    """
    y_prob = model.predict(test_data)
    y_pred = y_prob.argmax(axis=1)

    y_true = []
    y_true_one_hot = []

    for _, labels in test_data:
        y_true.extend(np.argmax(labels, axis=1))
        y_true_one_hot.append(labels)

    y_true = np.array(y_true)
    y_true_one_hot = np.concatenate(y_true_one_hot)

    return y_prob, y_pred, y_true, y_true_one_hot



def plot_classification_report(y_true, y_pred, target_names, figsize=(8, 4), cmap=plt.cm.Blues, save_pdf=None):
    """
    This function plots the classification report as a table-like plot.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        target_names (list): List of class labels.
        figsize (tuple): Optional. Size of the plot (width, height).
        cmap (str): Optional. Color map for the heatmap.
        save_pdf (str or None): Optional. If provided, the plot will be saved as a PDF with the given filename.

    Returns:
        None

    Example:
        plot_classification_report(y_true, y_pred, target_names=class_names)

    """
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, digits=4)
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=figsize)
    sns.heatmap(report_df, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Classification Report")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.tight_layout()

    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(8, 6), save_pdf=None):
    """
    This function plots the confusion matrix.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        classes (list): List of class labels.
        normalize (bool): If True, the confusion matrix will be normalized.
        title (str): Title for the plot.
        cmap: Color map for the plot.
        figsize (tuple): Optional. Size of the plot (width, height).
        save_pdf (str or None): Optional. If provided, the plot will be saved as a PDF with the given filename.

    Returns:
        None

    Example:
        plot_confusion_matrix(y_true, y_pred, classes=class_names)
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if title:
        plt.title(title)
    plt.tight_layout()

    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
        plt.close()
    else:
        plt.show()


def plot_multiclass_roc(y_true_one_hot, y_prob, class_names, lw=2, figsize=(10, 8), save_pdf=None):
    """
    Plot multiclass ROC curves along with micro and macro averages.

    Parameters:
        y_true_one_hot (array-like): One-hot matrix of true labels.
        y_prob (array-like): Predicted probabilities for each class.
        class_names (list): List of class labels.
        lw (float): Line width for the ROC curves.
        figsize (tuple): Optional. Size of the plot (width, height).
        save_pdf (str or None): Optional. If provided, the plot will be saved as a PDF with the given filename.

    Returns:
        None

    Example usage:
        plot_multiclass_roc(y_true_one_hot, y_prob, class_names, figsize=(12, 10), save_pdf="multiclass_roc.pdf")
    """
    n_classes = len(class_names)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=figsize)
    styles = cycle(['-', '--', '-.', ':'])
    for i, style in zip(range(n_classes), styles):
        plt.plot(fpr[i], tpr[i], lw=lw, linestyle=style,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, linestyle=':',
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    # Plot macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], color='navy', lw=lw, linestyle=':',
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Extending the ROC Curve to Multi-Class')
    plt.legend(loc="lower right")

    if save_pdf:
        plt.savefig(save_pdf, format='pdf')
        plt.close()
    else:
        plt.show()





