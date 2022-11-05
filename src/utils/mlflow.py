"""
NAME:
    mlflow

DESCRIPTION:
    Allows setting up tracking with mlflow.
"""

import mlflow
import uuid
import torch
import subprocess
import os
import ntpath
from joblib import dump
import h5py
import globals.constants as const


def init_mlflow(tracking_uri, exp_name, artifact_uri):
    """
    Initializes a mlflow instance and creates/loads an experiment within this instance.

    Parameters:
        tracking_uri: tracking URI
        exp_name: name of experiment to track to
        artifact_uri: uri for tracking the models

    Returns:
        mlflow: mlflow instance related to the tracking URI
        exp_id: id of experiment
    """

    # Set Tracking URI
    mlflow.set_tracking_uri(tracking_uri)

    # Stop any active mlflow run
    mlflow.end_run()

    # Retrieve experiment id or create experiment if necessary
    exp_id = '0'
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment:
        exp_id = experiment.experiment_id
    else:
        exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_uri)

    # Set experiment status to "active"
    mlflow.set_experiment(exp_name)

    return mlflow, exp_id


def start_mlflow_run():
    """
    Starts a new mlflow run and prevents eventually occuring mlflow error.

    Parameters:
        None

    Returns:
        None
    """

    # Stop any active mlflow run
    const.mlflow.end_run()

    # Start new run
    const.mlflow.start_run(experiment_id=const.exp_id, run_name=str(uuid.uuid4()))


def track_model(model, path_addition, pytorch=True):
    """
    Saves a model to a file in the "src/tracking/models" directory.
    Logs the model to the active mlflow run.

    Parameters:
        model: pytorch or sklearn model
        path_addition: string that extends the path for saving the model
                       by some further information to better recognize the type of the model
        pytorch: boolean that indicates whether the passed model is a pytorch model or not
                 pytorch models are stored in a .pth file, others must be storable in a .joblib file

    Returns:
        None
    """

    run_id = const.mlflow.active_run().info.run_id

    path = "tracking/models/"
    if pytorch:
        path += str(run_id) + "_" + path_addition + ".pth"
        torch.save(model, path)
    else:
        path += str(run_id) + "_" + path_addition + ".joblib"
        dump(model, path)

    copy_file_to_master(path)

    const.mlflow.log_artifact(path)


def track_training_data():
    """
    Saves the training data to a file in the "tracking/training_data" directory.
    Logs the data to the active mlflow run.

    Parameters:
        None

    Returns:
        None
    """

    run_id = const.mlflow.active_run().info.run_id

    path = "tracking/training_data/" + str(run_id) + '_train_ds.h5'

    # Create new h5 file
    hf = h5py.File(path, 'w')

    # Create datasets
    hf.create_dataset('st', data=const.st_train.cpu())
    hf.create_dataset('U', data=const.U_train.cpu())

    # Write file and contents to disk
    hf.close()

    copy_file_to_master(path)

    const.mlflow.log_artifact(path)


def copy_file_to_master(file_path):
    """
    Copies a file to the master machine on which the mlflow tracking server is running.
    !! Requires a authorized ssh-connection !!

    Parameters:
        file_path: local file that should be tracked

    Returns:
        None
    """

    # Make directory of mlrun first if it does not exist
    directory = const.artifact_uri[7:]
    directory += os.path.splitext(ntpath.basename(file_path))[0]
    directory = directory.split('_')[0]
    command = 'ssh ' + const.ssh_alias + ' mkdir -p ' + directory
    subprocess.run([command], shell=True)

    # Create artifact directory inside mlrun directory
    command = 'ssh ' + const.ssh_alias + ' mkdir -p ' + directory + '/artifacts'
    subprocess.run([command], shell=True)

    # Copy file to master
    command = 'scp ' + file_path + ' ' + const.ssh_alias + ':' + directory + '/artifacts/'
    subprocess.run([command], shell=True)
