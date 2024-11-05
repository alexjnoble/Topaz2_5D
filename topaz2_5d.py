#!/usr/bin/env python3
#
# Author: Alex J. Noble, assisted by Claude 3.5 Sonnet & GPT4o, 2024 @SEMC, MIT License
#
# Topaz 2.5D
#
# This script extends Topaz (2D) to pick slices of 3D tomograms given tomograms and
# corresponding 3D coordinates. It extends the capabilities of Topaz by preprocessing
# tomograms, training models on 2D slices, & aggregating 2D predictions into 3D coordinates.
#
# Dependencies: Topaz
#               pip install matplotlib mrcfile numpy opencv-python pandas pillow scikit-learn scipy
#
# Topaz is distributed under the GPL-3.0 license. For details, see:
#   https://www.gnu.org/licenses/gpl-3.0.en.html
# Topaz source code and installation instructions: https://github.com/tbepler/topaz/
# Ensure compliance with license terms when obtaining and using Topaz.
__version__ = "1.0.0"

import os
import sys
import glob
import json
import math
import time
import random
import hashlib
import inspect
import logging
import mrcfile
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

global_verbosity = 1  # Default verbosity level

def print_and_log(message, level=logging.INFO):
    """
    Prints and logs a message with the specified level, including debug details for verbosity level 3.

    :param str message: The message to print and log.
    :param int level: The logging level for the message (e.g., logging.DEBUG).

    If verbosity is set to 3, the function logs additional details about the caller,
    including module name, function name, line number, and function parameters.

    This function writes logging information to the disk.
    """
    logger = logging.getLogger()

    if global_verbosity < 3:
        # Directly log the primary message with the specified level for verbosity less than 3
        logger.log(level, message)
    else:
        # Retrieve the caller's frame to get additional context for verbosity level 3
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        line_no = caller_frame.f_lineno
        module_name = caller_frame.f_globals["__name__"]

        # Skip logging debug information for print_and_log calls to avoid recursion
        if func_name != 'print_and_log':
            # Retrieve function parameters and their values for verbosity level 3
            args, _, _, values = inspect.getargvalues(caller_frame)
            args_info = ', '.join([f"{arg}={values[arg]}" for arg in args])

            # Construct the primary message with additional debug information
            detailed_message = f"{message} - Debug - Module: {module_name}, Function: {func_name}({args_info}), Line: {line_no}"
            logger.log(level, detailed_message)
        else:
            # For print_and_log function calls, log only the primary message
            logger.log(level, message)

def setup_logging(script_start_time, verbosity):
    """
    Sets up logging configuration for console and file output based on verbosity level.

    :param int script_start_time: Timestamp used for naming the log file.
    :param int verbosity: A value that determines the level of detail for log messages. Supports:
      - 0 for ERROR level messages only,
      - 1 for WARNING level and above,
      - 2 for INFO level and above,
      - 3 for DEBUG level and all messages, including detailed debug information.

    The function configures both a console handler and a file handler for logging,
    with messages formatted according to the specified verbosity level.
    """
    global global_verbosity
    global_verbosity = verbosity

    # Map verbosity to logging level
    levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    console_logging_level = levels.get(verbosity)

    log_filename = f"topaz2_5d_{script_start_time}.log"

    # Formatters for logging
    simple_formatter = logging.Formatter('%(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d, %(funcName)s)')

    # File handler for logging, always records at least INFO level
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)  # Default to INFO level for file logging
    file_handler.setFormatter(detailed_formatter)

    # Console handler for logging, respects the verbosity level set by the user
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logging_level)
    if verbosity < 3:
        console_handler.setFormatter(simple_formatter)
    else:
        console_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)  # Set file logging to DEBUG for verbosity 3

    # Configuring the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set logger to highest level to handle all messages
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def parse_args(script_start_time):
    """
    Parses command-line arguments.

    :param str script_start_time: Function start time, formatted as a string
    :return argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Topaz 2.5D: Particle picking 3D cryoET tomograms by training and predicting on tomogram slices. Tomograms should already be binned (typically downsampled 4-16 times).",
    formatter_class=argparse.RawDescriptionHelpFormatter)  # Preserves whitespace for better formatting

    # Input/Output Options
    io_group = parser.add_argument_group('\033[1mInput/Output Options\033[0m')
    io_group.add_argument("mode", choices=["train", "extract"], help="Mode to run the script in: train or extract")
    io_group.add_argument("-t", "--tomograms", nargs='+', required=True, help="Paths to the input tomograms (MRC files) or directories containing MRC files")
    io_group.add_argument("-c", "--coordinates", nargs='+', help="Paths to the input coordinates files (x, y, z for each particle on each new line) or directories containing .coords, .txt, or .csv files")
    io_group.add_argument("-o", "--output", default='output', help="Directory to save output coordinate files")
    io_group.add_argument("--model_name", default='particle1', help="Model file name")

    # General Options
    general_group = parser.add_argument_group('\033[1mGeneral Options\033[0m')
    general_group.add_argument("-s", "--slice_thickness", type=int, default=1, help="Thickness of each slice for training/prediction where slices are averaged (default: 1)")
    general_group.add_argument("-n", "--num_slices", nargs='+', type=int, default=[10], help="Number of slices to extend symmetrically vertically for training and extraction for each particle type. For training, make this less than the minimum particle radius. For extraction, make this the average particle radius.")
    general_group.add_argument("--scale", type=int, default=1, help="Rescaling factor for tomogram down/upsampling, used in topaz preprocess and extract (default: 1)")
    general_group.add_argument("-e", "--expected_particles", type=int, default=500, help="Expected number of particles per tomogram for training")
    general_group.add_argument("-T", "--test_split", type=float, default=0.2, help="Percentage of tomogram slices to use for testing; remaining are used for training (default: 0.2 = 20%%")
    general_group.add_argument("-w", "--num_workers", type=int, default=8, help="Number of worker threads to use for topaz train (default: 8)")

    # Preprocessing specific arguments
    topaz_preprocess_group = parser.add_argument_group('\033[1mPreprocessing Options\033[0m')
    topaz_preprocess_group.add_argument("--preprocess_dir", type=str, default="preprocessed", help="Directory for Topaz preprocessing output")
    topaz_preprocess_group.add_argument("--sample", type=int, default=1, help="Pixel sampling factor for model fit (default: 1)")
    topaz_preprocess_group.add_argument("--preprocess_workers", type=int, default=-1, help="Number of processes to use for parallel image downsampling (default: -1 = all)")
    topaz_preprocess_group.add_argument("--niters", type=int, default=100, help="Number of iterations to run for model fit (default: 100)")
    topaz_preprocess_group.add_argument("--disable_topaz_preprocess", action="store_true", help="Disable Topaz preprocessing. Normalize slices instead")

    # Training specific arguments
    topaz_train_group = parser.add_argument_group('\033[1mTraining Options\033[0m')
    topaz_train_group.add_argument("--device", type=int, default=0, help="Which GPU device to use, set to -1 to force CPU (default: 0)")
    topaz_train_group.add_argument("--format", choices=["auto", "coord", "csv", "star", "box"], default="auto", help="File format of the particle coordinates file (default: auto)")
    topaz_train_group.add_argument("--image_ext", help="Sets the image extension if loading images from directory. Should include '.' before the extension (default: find all extensions)")
    topaz_train_group.add_argument("--k_fold", type=int, help="Option to split the training set into K folds for cross validation (default: not used)")
    topaz_train_group.add_argument("--fold", type=int, default=0, help="When using K-fold cross validation, sets which fold is used as the heldout test set (default: 0)")
    topaz_train_group.add_argument("--radius", type=int, default=1, help="Pixel radius around particle centers to consider positive (default: 1)")
    topaz_train_group.add_argument("--method", choices=["PN", "GE-KL", "GE-binomial", "PU"], default="GE-binomial", help="Objective function to use for learning the region classifier (default: GE-binomial)")
    topaz_train_group.add_argument("--slack", type=float, help="Weight on GE penalty (default: 10 for GE-KL, 1 for GE-binomial)")
    topaz_train_group.add_argument("--autoencoder", type=float, default=0, help="Option to augment method with autoencoder. Weight on reconstruction error (default: 0)")
    topaz_train_group.add_argument("--l2", type=float, default=0, help="L2 regularizer on the model parameters (default: 0)")
    topaz_train_group.add_argument("--natural", action="store_true", help="Sample unbiasedly from the data to form minibatches")
    topaz_train_group.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate for the optimizer (default: 0.0002)")
    topaz_train_group.add_argument("--minibatch_size", type=int, default=128, help="Number of data points per minibatch (default: 128)")
    topaz_train_group.add_argument("--minibatch_balance", type=float, default=0.0625, help="Fraction of minibatch that is positive data points (default: 0.0625)")
    topaz_train_group.add_argument("--epoch_size", type=int, default=1000, help="Number of parameter updates per epoch (default: 1000)")
    topaz_train_group.add_argument("--num_epochs", type=int, default=10, help="Maximum number of training epochs (default: 10)")
    topaz_train_group.add_argument("--no-pretrained", action="store_true", help="Disable initializing model parameters from the pretrained parameters (default: pretrained is used)")
    topaz_train_group.add_argument("--model", default="resnet8", help="Model type to fit (default: resnet8)")
    topaz_train_group.add_argument("--units", type=int, default=32, help="Number of units model parameter (default: 32)")
    topaz_train_group.add_argument("--dropout", type=float, default=0.0, help="Dropout rate model parameter (default: 0.0)")
    topaz_train_group.add_argument("--bn", choices=["on", "off"], default="on", help="Use batch norm in the model (default: on)")
    topaz_train_group.add_argument("--pooling", default="none", help="Pooling method to use (default: none)")
    topaz_train_group.add_argument("--unit_scaling", type=int, default=2, help="Scale the number of units up by this factor every pool/stride layer (default: 2)")
    topaz_train_group.add_argument("--ngf", type=int, default=32, help="Scaled number of units per layer in generative model, only used if autoencoder > 0 (default: 32)")
    topaz_train_group.add_argument("--save_prefix", help="Path prefix to save trained models each epoch")
    topaz_train_group.add_argument("--test_batch_size", type=int, default=1, help="Batch size for calculating test set statistics (default: 1)")
    topaz_train_group.add_argument("--plotting_train_sample_rate", type=int, default=50, help="Training sample rate for plotting (ie. don't plot all metrics, only 1 out of every N) (default: 50)")

    # Extract specific arguments
    topaz_extract_group = parser.add_argument_group('\033[1mExtract Options\033[0m')
    topaz_extract_group.add_argument("-m", "--model_files", nargs='+', default=["resnet16_u64"], help="Model file paths or names of pretrained models (options: resnet8_u32, resnet8_u64, resnet16_u32, resnet16_u64) (default: resnet16_u64)")
    topaz_extract_group.add_argument("-r", "--particle_radii", nargs='+', type=int, default=[10], help="Particle radii for each model for extraction. Used by Topaz to remove duplicate particles in 2D and by Topaz 2.5D to remove duplicates in 3D.")
    topaz_extract_group.add_argument("--names", nargs='+', default=["particle1"], help="Names for each particle type")
    topaz_extract_group.add_argument("--threshold", type=float, default=0, help="Score quantile giving threshold at which to terminate region extraction (default: 0.5)")
    topaz_extract_group.add_argument("--assignment_radius", type=float, help="Maximum distance between prediction and labeled target allowed for considering them a match (default: same as extraction radius)")
    topaz_extract_group.add_argument("--min_radius", type=int, default=5, help="Minimum radius for region extraction when tuning radius parameter (default: 5)")
    topaz_extract_group.add_argument("--max_radius", type=int, default=100, help="Maximum radius for region extraction when tuning radius parameters (default: 100)")
    topaz_extract_group.add_argument("--step_radius", type=int, default=5, help="Grid size when searching for optimal radius parameter (default: 5)")
    topaz_extract_group.add_argument("--extract_workers", type=int, default=-1, help="Number of processes to use for extracting in parallel, 0 uses main process, -1 uses all CPUs (default: -1)")
    topaz_extract_group.add_argument("--targets", help="Path to file specifying particle coordinates. Used to find extraction radius that maximizes the AUPRC")

    # System and Program Options
    misc_group = parser.add_argument_group('\033[1mSystem and Program Options\033[0m')
    misc_group.add_argument("-C", "--cpus", type=int, default=os.cpu_count(), help="Number of CPUs to use for various processing steps. Default is the number of CPU cores available: %(default)s")
    misc_group.add_argument("-V", "--verbosity", type=int, default=1, help="Verbosity level (default: 1)")
    misc_group.add_argument("-v", "--version", action="version", help="Show version number and exit", version=f"Topaz 2.5D v{__version__}")

    args = parser.parse_args()

    setup_logging(script_start_time, args.verbosity)

    if args.mode == 'train':
        args.num_slices = args.num_slices[0] // 2
    else:
        args.num_slices = [slice // 2 for slice in args.num_slices]
    args.preprocess_dir = os.path.join(os.getcwd(), args.preprocess_dir)
    if args.mode == "extract":
        if len(args.model_files) != len(args.particle_radii) or \
           len(args.model_files) != len(args.num_slices) or \
           len(args.model_files) != len(args.names):
            raise ValueError("Number of models, radii, num_slices, and names must match")

    print_and_log(f"\nInput command: {' '.join(sys.argv)}", logging.DEBUG)

    return args

def time_diff(time_diff):
    """
    Convert the time difference to a human-readable format.

    :param float time_diff: The time difference in seconds.
    :return str: A formatted string indicating the time difference.
    """
    seconds_in_day = 86400
    seconds_in_hour = 3600
    seconds_in_minute = 60

    days, time_diff = divmod(time_diff, seconds_in_day)
    hours, time_diff = divmod(time_diff, seconds_in_hour)
    minutes, seconds = divmod(time_diff, seconds_in_minute)

    time_str = ""
    if days > 0:
        time_str += f"{int(days)} day{'s' if days != 1 else ''}, "
    if hours > 0 or days > 0:  # Show hours if there are any days
        time_str += f"{int(hours)} hour{'s' if hours != 1 else ''}, "
    if minutes > 0 or hours > 0 or days > 0:  # Show minutes if there are any hours or days
        time_str += f"{int(minutes)} minute{'s' if minutes != 1 else ''}, "
    time_str += f"{int(seconds)} second{'s' if seconds != 1 else ''}"

    return time_str

def get_file_list(paths, extensions):
    """
    Get a list of files with specified extensions from given paths or directories.

    :param list paths: List of file paths or directories
    :param list extensions: List of file extensions to look for (e.g., ['.mrc', '.coords'])
    :return list: List of file paths
    """
    file_list = []
    for path in paths:
        if os.path.isdir(path):
            for ext in extensions:
                file_list.extend(glob.glob(os.path.join(path, f'*{ext}')))
        elif os.path.isfile(path):
            if any(path.endswith(ext) for ext in extensions):
                file_list.append(path)
    return sorted(set(file_list))  # Remove duplicates and sort

def readmrc(mrc_path):
    """
    Read an MRC file and return its data as a NumPy array.

    :param str mrc_path: The file path of the MRC file to read.
    :return numpy.ndarray: The data of the MRC file as a NumPy array.
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        numpy_array = np.array(data)

    # Rotate the array 90 degrees around the y-axis and then -90 degrees around the new z-axis (old x-axis)
    rotated_array = np.rot90(numpy_array, k=1, axes=(0, 2))
    rotated_array = np.rot90(rotated_array, k=-1, axes=(0, 1))

    return rotated_array

def writemrc(mrc_path, numpy_array, voxelsize=1.0):
    """
    Write a 3D NumPy array as an MRC file with specified voxel size.

    :param str mrc_path: The file path of the MRC file to write.
    :param numpy.ndarray numpy_array: The 3D NumPy array to be written.
    :param float voxelsize: The voxel size in Angstroms, assumed equal for all dimensions.
    :raises ValueError: If input numpy_array is not 2D or 3D.
    """
    if numpy_array.ndim not in [2, 3]:
        raise ValueError("Input array must be 2D or 3D")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(numpy_array)
        mrc.voxel_size = (voxelsize, voxelsize, voxelsize)
        mrc.update_header_from_data()
        mrc.update_header_stats()

def read_coordinates(coord_path):
    """
    Read coordinates from a file and return as a list of tuples.
    This function automatically detects space, tab, and comma delimiters.
    If the file doesn't exist, return an empty list.

    :param str coord_path: Path to the coordinate file.
    :return list: List of (x, y, z) coordinates, or an empty list if file doesn't exist.
    """
    if not os.path.exists(coord_path):
        return []
    coordinates = []
    with open(coord_path, 'r') as file:
        for line in file:
            # Try splitting with each delimiter
            for delimiter in [' ', '\t', ',']:
                parts = line.strip().split(delimiter)
                if len(parts) == 3:  # Check if there are 3 parts
                    x, y, z = map(float, parts)
                    coordinates.append((x, y, z))
                    break  # Move to the next line if successful
    return coordinates

def write_coordinates(coord_path, coordinates):
    """
    Write 3D coordinates to a file.

    :param str coord_path: Path to the coordinate file.
    :param list coordinates: List of (x, y, z) coordinates.
    """
    with open(coord_path, 'w') as file:
        file.writelines(f"{x} {y} {z}\n" for x, y, z in coordinates)

def write_extended_coordinates(coord_path, coordinates):
    """
    Write extended 2D coordinates to a file in Topaz format.
    Adds a header if it doesn't exist.

    :param str coord_path: Path to the coordinate file.
    :param list coordinates: List of tuples (image_name, x, y).
    """
    header = "image_name\tx_coord\ty_coord\n"

    mode = 'a' if (os.path.exists(coord_path) and open(coord_path, 'r').readline() == header) else 'w'

    with open(coord_path, mode) as file:
        if mode == 'w':
            file.write(header)
        file.writelines(f"{image_name}\t{x}\t{y}\n" for image_name, x, y in coordinates)

def extend_coordinates(coords, num_slices):
    """
    Extend coordinates upward and downward by a specified number of slices.

    :param list coords: List of (x, y, z) coordinates.
    :param int num_slices: Number of slices to extend.
    :return list: Extended list of coordinates.
    """
    # Extends each (x, y, z) in coords by +/- num_slices along the z-axis.
    return [(x, y, z + i) for x, y, z in coords for i in range(-num_slices, num_slices + 1)]

def get_unique_prefix_length(filenames):
    """
    Determine the minimum length of prefix needed to uniquely identify each filename.

    :param list filenames: List of filenames to analyze
    :return int: Minimum length of unique prefix
    """
    max_len = max(len(f) for f in filenames)
    for i in range(1, max_len + 1):
        prefixes = set(f[:i] for f in filenames)
        if len(prefixes) == len(filenames):
            return i
    return max_len

def strip_extensions(filename):
    """
    Strip all extensions from a filename.

    :param str filename: The filename to strip
    :return str: Filename without extensions
    """
    return filename.split('.')[0]

def match_files(tomograms, coordinates):
    """
    Match tomogram files with coordinate files based on unique prefixes.

    :param list tomograms: List of tomogram file paths
    :param list coordinates: List of coordinate file paths (can be empty)
    :return tuple: Matched pairs, unmatched tomograms, unmatched coordinates
    """
    tomo_names = [strip_extensions(os.path.basename(t)) for t in tomograms]

    if not coordinates:
        return [], tomograms, []

    coord_names = [strip_extensions(os.path.basename(c)) for c in coordinates]

    # Create a mapping of stripped names to original filenames
    tomo_dict = {name: t for name, t in zip(tomo_names, tomograms)}
    coord_dict = {name: c for name, c in zip(coord_names, coordinates)}

    matched_pairs = []
    unmatched_tomos = []
    unmatched_coords = []

    # Match files based on exact stripped names
    for name in set(tomo_names) & set(coord_names):
        matched_pairs.append((tomo_dict[name], coord_dict[name]))

    for name in set(tomo_names) - set(coord_names):
        unmatched_tomos.append(tomo_dict[name])

    for name in set(coord_names) - set(tomo_names):
        unmatched_coords.append(coord_dict[name])

    return matched_pairs, unmatched_tomos, unmatched_coords

def create_preprocessing_metadata(tomograms, preprocess_dir, disable_topaz_preprocess=False):
    """
    Create metadata file for preprocessed tomograms.

    :param list tomograms: List of tomogram file paths
    :param str preprocess_dir: Directory where preprocessed files are stored
    """
    metadata = {
        "tomograms": [
            {
                "filename": os.path.basename(tomo),
                "hash": compute_file_hash(tomo)
            } for tomo in tomograms
        ],
        "preprocessing_date": time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "topaz_version": __version__,
        "preprocessing_method": "simple_normalization" if disable_topaz_preprocess else "topaz_preprocess"
    }

    with open(os.path.join(preprocess_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def compute_file_hash(filepath):
    """
    Compute SHA256 hash of a file.

    :param str filepath: Path to the file
    :return str: Hexadecimal digest of the file hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_preprocessing_status(tomograms, preprocess_dir):
    """
    Check if preprocessing is needed based on existing metadata.

    :param list tomograms: List of tomogram file paths
    :param str preprocess_dir: Directory where preprocessed files are stored
    :return bool: True if preprocessing is needed, False otherwise
    """
    if not os.path.exists(preprocess_dir):
        return True

    metadata_file = os.path.join(preprocess_dir, "preprocessing_metadata.json")
    if not os.path.exists(metadata_file):
        return True

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    existing_tomos = {tomo["filename"]: tomo["hash"] for tomo in metadata["tomograms"]}

    # Compute hashes in parallel
    with ProcessPoolExecutor() as executor:
        future_to_tomo = {executor.submit(compute_file_hash, tomo): os.path.basename(tomo) for tomo in tomograms}
        current_tomos = {}
        for future in as_completed(future_to_tomo):
            tomo_name = future_to_tomo[future]
            try:
                hash_value = future.result()
                current_tomos[tomo_name] = hash_value
            except Exception as exc:
                print(f'Generating hash for {tomo_name} generated an exception: {exc}')

    return existing_tomos != current_tomos

def prompt_user_for_preprocessing(tomograms, preprocess_dir, disable_topaz_preprocess=False):
    """
    Check if preprocessing is needed and proceed accordingly.

    :param list tomograms: List of tomogram file paths
    :param str preprocess_dir: Directory where preprocessed files are stored
    :return bool: True if preprocessing is needed or user confirms preprocessing, False if user aborts
    """
    print_and_log("Preprocessing status check...")
    preprocessing_needed = check_preprocessing_status(tomograms, preprocess_dir)

    if preprocessing_needed and not os.path.exists(preprocess_dir):
        print_and_log(f"Preprocessing is needed. Proceeding with {'simple normalization' if disable_topaz_preprocess else 'Topaz preprocessing'}.")
        return True
    elif preprocessing_needed:
        while True:
            choice = input(f"Mismatch detected between current tomograms and preprocessed data.\nOptions:\n1. Reprocess all files using {'simple normalization' if disable_topaz_preprocess else 'Topaz preprocessing'}\n2. Show more information\n3. Abort operation\nEnter your choice (1/2/3): ").strip()
            if choice == '1':
                return True
            elif choice == '2':
                show_preprocessing_info(tomograms, preprocess_dir, force_print=True)
            elif choice == '3':
                print_and_log("Operation aborted by user.")
                sys.exit(0)
            else:
                print_and_log("Invalid choice. Please enter 1, 2, or 3.")
    else:
        print_and_log("Preprocessed data is up-to-date. No reprocessing needed.")
        return False

def show_preprocessing_info(tomograms, preprocess_dir, force_print=False):
    """
    Show detailed information about preprocessing status.

    :param list tomograms: List of tomogram file paths
    :param str preprocess_dir: Directory where preprocessed files are stored
    :param bool force_print: If True, print information regardless of verbosity level
    """
    info = ["\nDetailed preprocessing information:"]

    metadata_file = os.path.join(preprocess_dir, "preprocessing_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        info.append("Existing preprocessed data:")
        info.append(f"Preprocessing date: {metadata['preprocessing_date']}")
        info.append(f"Topaz 2.5D version: {metadata['topaz_version']}")
        info.append("Tomograms:")
        for tomo in metadata["tomograms"]:
            info.append(f"  - {tomo['filename']}")
    else:
        info.append("No existing preprocessing metadata found.")

    info.append("\nCurrent tomograms:")
    for tomo in tomograms:
        info.append(f"  - {os.path.basename(tomo)}")

    info.append("\nDebug information:")
    info.append("Tomogram files:")
    for tomo in tomograms:
        info.append(f"  - {tomo}")

    if force_print or global_verbosity >= 2:
        for line in info:
            print(line)
    else:
        for line in info:
            print_and_log(line, level=logging.DEBUG)

def average_slices_per_tomogram(tomogram_paths):
    """
    Calculate the average number of slices per tomogram for the given list of tomogram file paths.

    :param list tomogram_paths: List of file paths to the tomogram .mrc files
    :return float: Average number of slices per tomogram
    """
    slice_counts = []
    for file_path in tomogram_paths:
        try:
            with mrcfile.open(file_path, mode='r') as mrc:
                # Get the number of slices (z-dimension) for this tomogram
                num_slices = mrc.header.nz
                slice_counts.append(num_slices)
        except Exception as e:
            print_and_log(f"Error reading {file_path}: {str(e)}")

    if not slice_counts:
        print_and_log("No valid tomograms found. Cannot calculate average slices.")
        return 0  # or some default value

    average_slices = np.mean(slice_counts)
    return average_slices

def slice_tomogram(tomogram, slice_thickness=1):
    """
    Slice a 3D tomogram into 2D slices by averaging over the specified thickness.

    :param numpy.ndarray tomogram: The 3D tomogram as a NumPy array.
    :param int slice_thickness: The thickness of each slice to average over.
    :return list_of_numpy_arrays: A list of 2D slices as NumPy arrays.
    """
    slices = []
    for i in range(0, tomogram.shape[2], slice_thickness):
        if i + slice_thickness <= tomogram.shape[2]:
            # Average over the slice_thickness to create a single 2D slice
            slice_2d = np.mean(tomogram[:, :, i:i+slice_thickness], axis=2)
            slices.append(slice_2d)
    return slices

def process_tomogram(args):
    """
    Helper function to process a single tomogram for parallel execution.

    :param tuple args: (tomogram_path, coordinates, unstack_dir, slice_thickness, num_slices, mark_coords)
    :return tuple: (slices, extended_coords, tomogram_name)
    """
    tomogram_path, coordinates, unstack_dir, slice_thickness, num_slices, mark_coords = args

    tomogram = readmrc(tomogram_path)

    print_and_log(f"Slicing tomogram: {tomogram_path}")
    slices = slice_tomogram(tomogram, slice_thickness)

    tomogram_name = os.path.basename(tomogram_path).split('.')[0]

    if coordinates:
        print_and_log(f"Extending coordinates for: {tomogram_path}")
        extended_coords = extend_coordinates(coordinates, num_slices)
    else:
        extended_coords = []

    return slices, extended_coords, tomogram_name

def slice_and_write_tomograms(tomo_coord_dict, unstack_dir, slice_thickness, num_slices, mark_coords=False, max_workers=None):
    """
    Slices tomograms and writes them to disk, optionally marking coordinate points on the slices.

    :param dict tomo_coord_dict: Dictionary mapping tomogram paths to lists of coordinates (or tomogram paths only for extract mode).
    :param str unstack_dir: Directory to save the tomogram slices.
    :param int slice_thickness: Thickness of each slice to average over.
    :param int num_slices: Number of slices to extend coordinates symmetrically vertically.
    :param bool mark_coords: If True, mark coordinate points on the slices (only if coordinates are provided).
    :param int max_workers: Maximum number of worker processes to use. If None, it uses the number of CPU cores.
    :return tuple: (all_slices, all_extended_coords) or just all_slices if no coordinates are provided.
    """
    all_slices = []
    all_extended_coords = []
    slice_index = 0

    # Check if we're in extract mode (no coordinates)
    extract_mode = isinstance(next(iter(tomo_coord_dict.values())), str)

    # Prepare arguments for parallel processing
    if extract_mode:
        args_list = [(tomo, None, unstack_dir, slice_thickness, num_slices, mark_coords) 
                     for tomo in tomo_coord_dict]
    else:
        args_list = [(tomo, coords, unstack_dir, slice_thickness, num_slices, mark_coords) 
                     for tomo, coords in tomo_coord_dict.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_tomogram, args_list))

    for slices, extended_coords, tomogram_name in results:
        for j, slice_2d in enumerate(slices):
            slice_name = f"{tomogram_name}_slice_{slice_index:05d}"
            slice_path = f"{unstack_dir}/{slice_name}.mrc"
            os.makedirs(unstack_dir, exist_ok=True)

            if mark_coords and extended_coords:
                # Normalize slice to 0-255 range
                slice_min, slice_max = slice_2d.min(), slice_2d.max()
                normalized_slice = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)

                # Convert normalized slice to PIL Image
                slice_image = Image.fromarray(normalized_slice).convert("RGB")
                draw = ImageDraw.Draw(slice_image)

                # Draw coordinate points
                for x, y, z in extended_coords:
                    if int(z) == j:
                        draw.ellipse([x-1, y-1, x+1, y+1], fill="white", outline="white")

                # Convert back to numpy array
                marked_slice = np.array(slice_image)[:,:,0]  # Take only one channel
                writemrc(slice_path, marked_slice)
            else:
                writemrc(slice_path, slice_2d)

            all_slices.append(slice_path)
            if extended_coords:
                for x, y, z in extended_coords:
                    if int(z) == j:
                        all_extended_coords.append((slice_name, x, y))
            slice_index += 1

    return (all_slices, all_extended_coords) if all_extended_coords else all_slices

def run_topaz_preprocess(input_dir, output_dir, scale, sample, num_workers, device, niters):
    """
    Run topaz preprocess command with additional options.

    :param str input_dir: Directory of input files.
    :param str output_dir: Directory to save output files.
    :param int scale: Rescaling factor for image downsampling.
    :param int num_workers: Number of processes to use for parallel image downsampling.
    :param int pixel_sampling: Pixel sampling factor for model fit.
    :param int niters: Number of iterations to run for model fit.
    """
    command = [
        "topaz", "preprocess",
        os.path.join(input_dir, "*.mrc"),
        "--verbose",
        "--scale", str(scale),
        "--sample", str(sample),
        "--num-workers", str(num_workers),
        "--format", "mrc",
        "--device", str(device),
        "--niters", str(niters),
        "--destdir", output_dir
    ]
    cmd = " ".join(command)
    print_and_log(cmd)
    subprocess.run(cmd, check=True, shell=True)

def normalize_slice(args):
    """
    Normalize a single slice (0 mean, 1 std) and save it.

    :param tuple args: (slice_path, output_path)
    """
    slice_path, output_path = args
    with mrcfile.open(slice_path, permissive=True) as mrc:
        slice_data = mrc.data

    normalized_slice = (slice_data - np.mean(slice_data)) / np.std(slice_data)

    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(normalized_slice.astype(np.float32))

def normalize_slices(slice_paths, output_dir, max_workers):
    """
    Normalize slices in parallel and save them to the output directory.

    :param list slice_paths: List of paths to input slices
    :param str output_dir: Directory to save normalized slices
    :param int max_workers: Maximum number of worker processes to use
    """
    os.makedirs(output_dir, exist_ok=True)

    args_list = [(slice_path, os.path.join(output_dir, os.path.basename(slice_path))) 
                 for slice_path in slice_paths]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(normalize_slice, args_list))

def split_dataset(image_dir, particles_file, test_split, seed=None):
    """
    Implement train_test_split functionality directly based on the percentage of data for testing,
    with random selection of images.
    This function replaces topaz train_test_split because of a pandas incompatibility.

    :param str image_dir: Directory of images.
    :param str particles_file: File with particle coordinates.
    :param float test_split: Percentage of data to use for testing.
    :param int seed: Random seed for reproducibility. If None, a random seed will be generated.
    """
    def create_image_list(images, filename):
        with open(os.path.join(image_dir, filename), 'w') as f:
            f.write("image_name\tpath\n")
            for img in images:
                full_path = os.path.join(image_dir, f"{img}.mrc")
                f.write(f"{img}\t{full_path}\n")

    # If no seed is provided, generate a random one
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)

    # Get list of image files without extensions and read particles file
    image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.mrc')]
    particles_df = pd.read_csv(particles_file, sep='\t')

    # Split image files into train and test sets
    train_images, test_images = train_test_split(image_files, test_size=test_split, random_state=seed)

    # Create train and test dataframes, and save them
    train_df = particles_df[particles_df['image_name'].isin(train_images)]
    test_df = particles_df[particles_df['image_name'].isin(test_images)]
    train_df.to_csv(os.path.join(image_dir, "particles_train.txt"), sep='\t', index=False)
    test_df.to_csv(os.path.join(image_dir, "particles_test.txt"), sep='\t', index=False)

    create_image_list(train_images, "image_list_train.txt")
    create_image_list(test_images, "image_list_test.txt")

    print_and_log(f"Train: {len(train_images)} images, saved as particles_train.txt and image_list_train.txt")
    print_and_log(f"Test: {len(test_images)} images, saved as particles_test.txt and image_list_test.txt")

def run_topaz_train(train_images_dir, train_targets_file, test_images_file, test_targets_file, model_save_prefix, test_train_curve, expected_particles, num_workers, device, **kwargs):
    """
    Run Topaz train command with additional options.

    :param str train_images_dir: Directory of training images.
    :param str train_targets_file: File with training targets.
    :param str model_save_prefix: Prefix for saving the trained model.
    :param int expected_particles: Expected number of particles per image.
    :param int num_workers: Number of worker threads to use.
    :param int device: Device to use for training.
    :param str test_images: Directory of test images.
    :param str test_targets: File with test targets.
    """
    command = [
        "topaz", "train",
        "--num-particles", str(expected_particles),
        "--num-workers", str(num_workers),
        "--train-images", train_images_dir,
        "--train-targets", train_targets_file,
        "--test-images", test_images_file,
        "--test-targets", test_targets_file,
        "--save-prefix", model_save_prefix,
        "--device", str(device),
        "--output", str(test_train_curve)
    ]

    # Add optional arguments from kwargs
    optional_args = {
        "--format": kwargs.get("format"),
        "--image-ext": kwargs.get("image_ext"),
        "--k-fold": kwargs.get("k_fold"),
        "--fold": kwargs.get("fold"),
        "--pi": kwargs.get("pi"),
        "--radius": kwargs.get("radius"),
        "--method": kwargs.get("method"),
        "--slack": kwargs.get("slack"),
        "--autoencoder": kwargs.get("autoencoder"),
        "--l2": kwargs.get("l2"),
        "--learning-rate": kwargs.get("learning_rate"),
        "--minibatch-size": kwargs.get("minibatch_size"),
        "--minibatch-balance": kwargs.get("minibatch_balance"),
        "--epoch-size": kwargs.get("epoch_size"),
        "--num-epochs": kwargs.get("num_epochs"),
        "--model": kwargs.get("model"),
        "--units": kwargs.get("units"),
        "--dropout": kwargs.get("dropout"),
        "--bn": kwargs.get("bn"),
        "--pooling": kwargs.get("pooling"),
        "--unit-scaling": kwargs.get("unit_scaling"),
        "--ngf": kwargs.get("ngf"),
        "--save-prefix": kwargs.get("save_prefix"),
        "--test-batch-size": kwargs.get("test_batch_size")
    }

    # Add boolean flags if they are set
    if not kwargs.get("no_pretrained"):
        command.append("--pretrained")
    if kwargs.get("natural"):
        command.append("--natural")

    # Include the optional arguments in the command
    for arg, value in optional_args.items():
        if value is not None:
            command.extend([arg, str(value)])

    cmd = " ".join(command)
    print_and_log(cmd)
    subprocess.run(cmd, check=True, shell=True)

def run_topaz_extract(model_file, input_dir, output_file, particle_radius, scale_factor=1, **kwargs):
    """
    Run Topaz extract command and return the results as a DataFrame.

    :param str model_file: Path to the trained model file or name of a pretrained model.
    :param str input_dir: Directory containing input image files.
    :param str output_file: Path to save the output coordinates.
    :param int particle_radius: Radius of particles to extract.
    :param int scale_factor: Scaling factor for the coordinates (default: 1).
    :param kwargs: Additional keyword arguments for Topaz extract.
    :return pd.DataFrame: DataFrame containing extracted particle coordinates and scores.
    """
    # Base command with required arguments
    command = [
        "topaz", "extract",
        "--model", model_file,
        "--radius", str(particle_radius),
        "--up-scale", str(scale_factor),
        "--output", output_file
    ]

    # Add optional arguments from kwargs
    optional_args = {
        "--threshold": kwargs.get("threshold"),
        "--assignment-radius": kwargs.get("assignment_radius"),
        "--min-radius": kwargs.get("min_radius"),
        "--max-radius": kwargs.get("max_radius"),
        "--step-radius": kwargs.get("step_radius"),
        "--num-workers": kwargs.get("extract_workers"),
        "--targets": kwargs.get("targets"),
        "--device": kwargs.get("device")
    }

    # Include the optional arguments in the command
    for arg, value in optional_args.items():
        if value is not None:
            command.extend([arg, str(value)])

    # Create a list of .mrc files
    mrc_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mrc"):
                mrc_files.append(os.path.join(root, file))

    # Convert the command list to a string
    cmd = " ".join(command)
    print_and_log(cmd + " [file list truncated]")
    
    # Run the command with input redirection (allows for very long lists of files to be picked without shell errors)
    try:
        process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, text=True)
        for file in mrc_files:
            process.stdin.write(file + "\n")
        process.stdin.close()
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    except subprocess.CalledProcessError as e:
        print_and_log(f"Error running Topaz extract: {e}")
        raise

def read_predictions(pred_path):
    """
    Read predictions from a file and return as a pandas DataFrame.

    :param str pred_path: Path to the predictions file.
    :return pd.DataFrame: DataFrame with columns 'tomogram', 'slice', 'x', 'y', 'score'.
    """
    df = pd.read_csv(pred_path, sep='\t')
    df[['tomogram', 'slice']] = df['image_name'].str.rsplit('_slice_', n=1, expand=True)
    df['slice'] = df['slice'].astype(int)
    df = df.rename(columns={'x_coord': 'x', 'y_coord': 'y'})
    return df[['tomogram', 'slice', 'x', 'y', 'score']]

def aggregate_predictions(df, num_slices, particle_radius, slice_thickness=1, xy_tolerance=2):
    """
    Aggregate 2D predictions into 3D coordinates and remove duplicates.

    :param pd.DataFrame df: DataFrame with columns 'tomogram', 'slice', 'x', 'y', 'score'.
    :param int or list num_slices: Number of slices to consider for each particle. Can be a single int or a list of ints.
    :param int particle_radius: Radius of particles for duplicate removal.
    :param int slice_thickness: Thickness of each slice used in prediction (default: 1).
    :param float xy_tolerance: Tolerance for x and y coordinate similarity in pixels (default: 2).
    :return pd.DataFrame: Aggregated and deduplicated 3D coordinates with scores.
    """
    # If num_slices is a list, use the first (and only) value
    if isinstance(num_slices, list):
        num_slices = num_slices[0]
    aggregated_coords = []

    for tomogram in df['tomogram'].unique():
        tomogram_df = df[df['tomogram'] == tomogram].sort_values('slice')

        # Create bins for x and y coordinates
        tomogram_df['x_bin'] = (tomogram_df['x'] // xy_tolerance).astype(int)
        tomogram_df['y_bin'] = (tomogram_df['y'] // xy_tolerance).astype(int)

        # Group predictions by their x and y bins
        grouped = tomogram_df.groupby(['x_bin', 'y_bin'])

        for _, group in grouped:
            if len(group) < 2:  # Ignore single predictions
                continue

            # Check if predictions are within the required number of slices
            slice_range = group['slice'].max() - group['slice'].min() + 1
            if slice_range <= num_slices:
                # Calculate average coordinates and maximum score
                avg_x = group['x'].mean()
                avg_y = group['y'].mean()
                avg_z = (group['slice'].min() + slice_range / 2) * slice_thickness
                max_score = group['score'].max()

                aggregated_coords.append({
                    'tomogram': tomogram,
                    'x': avg_x,
                    'y': avg_y,
                    'z': avg_z,
                    'score': max_score
                })

    # Convert to DataFrame
    result_df = pd.DataFrame(aggregated_coords)

    # Remove duplicates
    final_coords = []
    for tomogram in result_df['tomogram'].unique():
        tomogram_df = result_df[result_df['tomogram'] == tomogram].copy()
        coords = tomogram_df[['x', 'y', 'z']].values

        # Calculate pairwise distances
        distances = pdist(coords)
        distance_matrix = squareform(distances)

        # Find pairs of particles that are too close
        close_pairs = np.argwhere(distance_matrix < 2 * particle_radius)
        close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]  # Keep only upper triangle

        # Remove lower scoring particle from each pair
        to_remove = set()
        for i, j in close_pairs:
            if tomogram_df.iloc[i]['score'] >= tomogram_df.iloc[j]['score']:
                to_remove.add(j)
            else:
                to_remove.add(i)

        # Keep particles that are not in the to_remove set
        keep_indices = [i for i in range(len(tomogram_df)) if i not in to_remove]
        final_coords.extend(tomogram_df.iloc[keep_indices].to_dict('records'))

    return pd.DataFrame(final_coords)

def run_topaz_extract_wrapper(args):
    """
    Wrapper function to run Topaz extract for a single model and process the results.

    This function unpacks the arguments, runs Topaz extract, aggregates the predictions,
    and adds particle type information to the results.

    :param tuple args: A tuple containing the following elements:
        - model (str): Path to the trained model file or name of a pretrained model.
        - radius (int): Radius of particles to extract.
        - num_slice (int): Number of slices to consider for this particle type.
        - name (str): Name of the particle type.
        - preprocess_dir (str): Directory containing preprocessed images.
        - scale (int): Scaling factor for coordinates.
        - workers_per_extract (int): Number of workers to use for this extraction.
        - kwargs (dict): Additional keyword arguments for Topaz extract.

    :return pd.DataFrame: DataFrame containing the processed and aggregated predictions
                          for the current model, including particle type information.

    :raises subprocess.CalledProcessError: If the Topaz extract command fails.
    :raises FileNotFoundError: If the predictions file is not created.
    :raises pd.errors.EmptyDataError: If the predictions file is empty or cannot be read.
    """
    model, radius, num_slice, name, preprocess_dir, scale, workers_per_extract, kwargs = args
    print_and_log(f"Extracting particles for model: {name}")
    predictions_file = f"{kwargs['output']}/predicted/{name}_predicted_coordinates.txt"

    # Update kwargs with the correct number of workers
    kwargs_copy = kwargs.copy()
    kwargs_copy['extract_workers'] = workers_per_extract

    # Run Topaz extract for current model
    predictions = run_topaz_extract(model, preprocess_dir, predictions_file, radius, scale, **kwargs_copy)

    # Aggregate predictions for current model
    predictions_df = read_predictions(predictions_file)
    predictions = aggregate_predictions(predictions_df, num_slice, radius, kwargs.get('slice_thickness', 1))

    # Add particle type information
    predictions['type'] = name
    predictions['type_index'] = kwargs['type_index']

    return predictions

def run_competitive_extraction(tomograms, models, radii, num_slices, names, preprocess_dir, scale, **kwargs):
    """
    Run Topaz extract for multiple models and aggregate results for competitive picking.
    This function parallelizes the extraction process based on available CPU cores.

    :param list tomograms: List of tomogram file paths.
    :param list models: List of model file paths or names.
    :param list radii: List of particle radii for each model.
    :param list num_slices: List of number of slices to consider for each model.
    :param list names: List of names for each particle type.
    :param str preprocess_dir: Directory containing preprocessed images.
    :param int scale: Scaling factor for coordinates.
    :param kwargs: Additional keyword arguments for Topaz extract.
    :return pd.DataFrame: Combined and aggregated predictions from all models.
    """
    total_cpus = kwargs.get('cpus', os.cpu_count())
    workers_per_extract = min(8, total_cpus)
    max_concurrent_extracts = math.floor(total_cpus / workers_per_extract)

    args_list = [
        (model, radius, num_slice, name, preprocess_dir, scale, workers_per_extract, {**kwargs, 'type_index': i})
        for i, (model, radius, num_slice, name) in enumerate(zip(models, radii, num_slices, names))
    ]

    all_predictions = []

    # Process extractions in batches
    for i in range(0, len(args_list), max_concurrent_extracts):
        batch = args_list[i:i+max_concurrent_extracts]

        with ProcessPoolExecutor(max_workers=max_concurrent_extracts) as executor:
            batch_results = list(executor.map(run_topaz_extract_wrapper, batch))

        all_predictions.extend(batch_results)

    # Combine predictions from all models
    return pd.concat(all_predictions, ignore_index=True)

def normalize_scores(particles):
    """
    Normalize scores within each particle type.

    :param pd.DataFrame particles: DataFrame containing particle information including scores and types.
    :return pd.DataFrame: DataFrame with added normalized scores.
    """
    for type_name in particles['type'].unique():
        mask = particles['type'] == type_name
        min_score = particles.loc[mask, 'score'].min()
        max_score = particles.loc[mask, 'score'].max()

        # Calculate normalized score for current particle type
        particles.loc[mask, 'normalized_score'] = (particles.loc[mask, 'score'] - min_score) / (max_score - min_score)

    return particles

def pick_for_tomogram(args):
    """
    Perform competitive picking for a single tomogram.

    :param tuple args: Tuple containing (tomogram_particles, radii)
    :return DataFrame: DataFrame of selected particles for this tomogram.
    """
    tomogram_particles, radii = args
    tomogram_particles = tomogram_particles.sort_values('normalized_score', ascending=False)
    coords = tomogram_particles[['x', 'y', 'z']].values

    to_keep = np.ones(len(tomogram_particles), dtype=bool)
    for i in range(len(tomogram_particles)):
        if to_keep[i]:
            distances = np.linalg.norm(coords[i+1:] - coords[i], axis=1)
            current_radius = radii[int(tomogram_particles.iloc[i]['type_index'])]
            other_radii = np.array([radii[int(idx)] for idx in tomogram_particles.iloc[i+1:]['type_index']])
            max_distances = current_radius + other_radii
            to_keep[i+1:] &= distances >= max_distances

    return tomogram_particles[to_keep]

def competitive_picking(particles, radii, max_workers=None):
    """
    Perform competitive picking among different particle types across all tomograms in parallel.

    :param pd.DataFrame particles: DataFrame containing normalized particle predictions for all tomograms.
    :param list radii: List of particle radii for each particle type.
    :param max_workers: Maximum number of worker processes to use. If None, it uses the number of CPU cores.
    :return pd.DataFrame: Final selected particles after competitive picking.
    """
    # Group particles by tomogram
    tomogram_groups = [group for _, group in particles.groupby('tomogram')]

    # Prepare arguments for parallel processing
    args_list = [(group, radii) for group in tomogram_groups]

    # Perform competitive picking in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        picked_particles = list(executor.map(pick_for_tomogram, args_list))

    # Combine results from all tomograms
    final_particles = pd.concat(picked_particles, ignore_index=True)

    return final_particles

def save_aggregated_coordinates(final_particles, threshold, output_dir):
    """
    Save aggregated 3D coordinates to a file.

    :param pd.DataFrame final_particles: DataFrame with columns 'tomogram', 'type', 'x', 'y', 'z', 'score', 'normalized_score'.
    :param float threshold: Topaz score threshold used previously to filter picks.
    :param str output_dir: Directory to save output files.
    """
    # Save aggregated_coordinates.txt with all information
    output_file = os.path.join(output_dir, "aggregated_coordinates.txt")
    final_particles.to_csv(output_file, sep='\t', index=False, 
                           columns=['tomogram', 'type', 'x', 'y', 'z', 'score', 'normalized_score'])

    # Save individual .coords files for each tomogram and particle type
    for tomogram in final_particles['tomogram'].unique():
        for particle_type in final_particles['type'].unique():
            mask = (final_particles['tomogram'] == tomogram) & (final_particles['type'] == particle_type)
            if mask.any():
                filename = os.path.join(output_dir, f"{tomogram}_{particle_type}.coords")
                final_particles.loc[mask, ['x', 'y', 'z']].to_csv(filename, sep=' ', header=False, index=False)

    print_and_log(f"All {len(final_particles)} coordinates and scores > {threshold} saved: '{output_file}'")
    print_and_log(f"Individual tomogram and particle type coords files saved in: {output_dir}")

def plot_training_metrics(file_path, train_sample_rate=1):
    """
    Read a Topaz training metrics file and plot the meaningful information.

    :param str file_path: Path to the Topaz training metrics file
    :param int train_sample_rate: Sample rate for training points (e.g., 10 means plot every 10th point)
    """
    # Read the metrics file
    df = pd.read_csv(file_path, sep='\t')

    # Create copies of the data to avoid SettingWithCopyWarning
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Convert 'auprc' and 'ge_penalty' to numeric, replacing any non-numeric values with NaN
    test_df.loc[:, 'auprc'] = pd.to_numeric(test_df['auprc'], errors='coerce')
    train_df.loc[:, 'ge_penalty'] = pd.to_numeric(train_df['ge_penalty'], errors='coerce')

    # Remove any rows with NaN values in 'auprc' or 'ge_penalty'
    test_df = test_df.dropna(subset=['auprc'])
    train_df = train_df.dropna(subset=['ge_penalty'])

    # Sample training data
    train_df = train_df.iloc[::train_sample_rate, :]

    # Set up the plot
    plt.figure(figsize=(20, 15))

    # Helper function to plot metrics
    def plot_metric(ax, metric, title, ylabel):
        ax.plot(train_df['iter'], train_df[metric], label=f'Train {metric.capitalize()}', color='blue', alpha=0.5)
        ax.scatter(test_df['iter'], test_df[metric], color='red', label=f'Test {metric.capitalize()}', zorder=5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add epoch numbers to x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(test_df['iter'])
        ax2.set_xticklabels(test_df['epoch'], fontweight='bold')
        ax2.set_xlabel('Epoch', fontweight='bold')

        # Make iteration ticks less prominent
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel('Iteration', fontsize=10)

    # Plot loss
    ax1 = plt.subplot(3, 2, 1)
    plot_metric(ax1, 'loss', 'Loss over Iterations', 'Loss')

    # Plot GE penalty (only for train)
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(train_df['iter'], train_df['ge_penalty'], color='blue', alpha=0.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('GE Penalty')
    ax2.set_title('GE Penalty over Iterations (Train)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Fix y-axis for GE Penalty plot
    ge_min, ge_max = train_df['ge_penalty'].min(), train_df['ge_penalty'].max()
    y_margin = (ge_max - ge_min) * 0.1  # Add 10% margin
    ax2.set_ylim(ge_min - y_margin, ge_max + y_margin)

    # Set y-ticks to ensure they are evenly spaced and not too cluttered
    num_ticks = 5  # You can adjust this number for more or fewer ticks
    y_ticks = np.linspace(ge_min, ge_max, num_ticks)
    ax2.set_yticks(y_ticks)

    # Format y-tick labels to be more readable
    ax2.set_yticklabels([f'{tick:.2e}' for tick in y_ticks])  # Scientific notation with 2 decimal places

    # Use a logarithmic scale if the range is very large
    if ge_max / ge_min > 1000:  # Adjust this threshold as needed
        ax2.set_yscale('log')
        ax2.set_ylim(max(ge_min, 1e-10), ge_max * 1.1)  # Avoid zero in log scale

    # Add epoch numbers to x-axis for GE Penalty plot
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(test_df['iter'])
    ax2_top.set_xticklabels(test_df['epoch'], fontweight='bold')
    ax2_top.set_xlabel('Epoch', fontweight='bold')

    # Make iteration ticks less prominent
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlabel('Iteration', fontsize=10)

    # Plot precision
    ax3 = plt.subplot(3, 2, 3)
    plot_metric(ax3, 'precision', 'Precision over Iterations', 'Precision')

    # Plot TPR (True Positive Rate)
    ax4 = plt.subplot(3, 2, 4)
    plot_metric(ax4, 'tpr', 'True Positive Rate over Iterations', 'True Positive Rate')

    # Plot FPR (False Positive Rate)
    ax5 = plt.subplot(3, 2, 5)
    plot_metric(ax5, 'fpr', 'False Positive Rate over Iterations', 'False Positive Rate')

    # Plot AUPRC (Area Under Precision-Recall Curve) - only for test data
    ax6 = plt.subplot(3, 2, 6)
    ax6.scatter(test_df['iter'], test_df['auprc'], color='red')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('AUPRC')
    ax6.set_title('Area Under Precision-Recall Curve (Test)')
    ax6.grid(True, linestyle='--', alpha=0.7)

    # Fix y-axis for AUPRC plot
    auprc_min = test_df['auprc'].min()
    auprc_max = test_df['auprc'].max()
    y_margin = (auprc_max - auprc_min) * 0.1  # Add 10% margin
    ax6.set_ylim(auprc_min - y_margin, auprc_max + y_margin)

    # Set y-ticks to ensure they are in order
    y_ticks = np.linspace(auprc_min, auprc_max, 10)
    ax6.set_yticks(y_ticks)
    ax6.set_yticklabels([f'{tick:.4f}' for tick in y_ticks])

    # Add epoch numbers to x-axis for AUPRC plot
    ax6_top = ax6.twiny()
    ax6_top.set_xlim(ax6.get_xlim())
    ax6_top.set_xticks(test_df['iter'])
    ax6_top.set_xticklabels(test_df['epoch'], fontweight='bold')
    ax6_top.set_xlabel('Epoch', fontweight='bold')

    # Make iteration ticks less prominent
    ax6.tick_params(axis='x', labelsize=8)
    ax6.set_xlabel('Iteration', fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('training_metrics_plot.png', dpi=300)
    plt.close()

    print_and_log(f"Training metrics plot: 'training_metrics_plot.png' ({train_sample_rate}-point sampling)")

def find_best_epoch(file_path, auprc_threshold=0.001, secondary_metric_weight=0.2):
    """
    Determine the best epoch based on the training metrics.

    :param str file_path: Path to the Topaz training metrics file
    :param float auprc_threshold: Threshold for considering AUPRC values as similar
    :param float secondary_metric_weight: Weight given to secondary metrics in case of similar AUPRC values
    :return tuple: (best_epoch_multi_metric, best_epoch_auprc_only)
    """
    # Read the metrics file
    df = pd.read_csv(file_path, sep='\t')

    # Filter for test data
    test_df = df[df['split'] == 'test'].copy()

    # Convert columns to numeric, replacing any non-numeric values with NaN
    numeric_columns = ['auprc', 'loss', 'fpr']
    for col in numeric_columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # Drop any rows with NaN values
    test_df.dropna(subset=numeric_columns, inplace=True)

    # If all rows were NaN, we can't proceed
    if test_df.empty:
        print_and_log("No valid numeric data found in the test set.")
        return None, None

    # Normalize secondary metrics
    test_df['norm_loss'] = (test_df['loss'] - test_df['loss'].min()) / (test_df['loss'].max() - test_df['loss'].min())
    test_df['norm_fpr'] = (test_df['fpr'] - test_df['fpr'].min()) / (test_df['fpr'].max() - test_df['fpr'].min())

    # Calculate a score that considers AUPRC and secondary metrics
    test_df['score'] = test_df['auprc'] - secondary_metric_weight * (test_df['norm_loss'] + test_df['norm_fpr'])

    # Find the best score (multi-metric approach)
    best_score = test_df['score'].max()
    best_epochs = test_df[test_df['score'] >= best_score - auprc_threshold]

    # If there are multiple 'best' epochs, choose the earliest one
    best_epoch_multi_metric = best_epochs['epoch'].min()

    # Find the best epoch based only on AUPRC
    best_epoch_auprc_only = test_df.loc[test_df['auprc'].idxmax(), 'epoch']

    # Print information for multi-metric approach
    print_and_log("Best epoch (multi-metric approach):")
    print_and_log(f"Epoch: {best_epoch_multi_metric}")
    print_and_log(f"AUPRC: {test_df.loc[test_df['epoch'] == best_epoch_multi_metric, 'auprc'].values[0]:.4f}")
    print_and_log(f"Loss: {test_df.loc[test_df['epoch'] == best_epoch_multi_metric, 'loss'].values[0]:.4f}")
    print_and_log(f"FPR: {test_df.loc[test_df['epoch'] == best_epoch_multi_metric, 'fpr'].values[0]:.4f}")

    # Print information for AUPRC-only approach
    print_and_log("\nBest epoch (AUPRC-only approach):")
    print_and_log(f"Epoch: {best_epoch_auprc_only}")
    print_and_log(f"AUPRC: {test_df.loc[test_df['epoch'] == best_epoch_auprc_only, 'auprc'].values[0]:.4f}")
    print_and_log(f"Loss: {test_df.loc[test_df['epoch'] == best_epoch_auprc_only, 'loss'].values[0]:.4f}")
    print_and_log(f"FPR: {test_df.loc[test_df['epoch'] == best_epoch_auprc_only, 'fpr'].values[0]:.4f}")

    return best_epoch_multi_metric, best_epoch_auprc_only

def main():
    """
    Main function: Determines mode, loops through processes, prints and logs each step.
    """
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))

    args = parse_args(start_time_formatted)
    global global_verbosity
    global_verbosity = args.verbosity

    unstack_dir = os.path.join(os.getcwd(), "unstacked")
    extended_coords_file = f"{unstack_dir}/extended_coordinates.txt"
    # Get file lists for tomograms and coordinates
    tomogram_files = get_file_list(args.tomograms, ['.mrc'])
    coordinate_files = get_file_list(args.coordinates, ['.coords', '.txt', '.csv']) if args.coordinates else []
    try:
        if args.mode == "train":
            print_and_log("Training on all tomograms")

            # Match tomograms with coordinate files
            matched_pairs, unmatched_tomos, unmatched_coords = match_files(tomogram_files, coordinate_files)

            # Create a dictionary of tomograms and their coordinates
            tomo_coord_dict = {}
            for tomo, coord in matched_pairs:
                tomo_coord_dict[tomo] = read_coordinates(coord)

            if unmatched_tomos:
                print_and_log("Warning: The following tomograms have no matching coordinate files:")
                for tomo in unmatched_tomos:
                    print_and_log(f"  - {os.path.basename(tomo)}")

                include_unpaired = input("Include unpaired tomograms in training? (y/n): ").strip().lower() == 'y'
                if include_unpaired:
                    print_and_log("Including unpaired tomograms in training")
                    for tomo in unmatched_tomos:
                        tomo_coord_dict[tomo] = []
                else:
                    print_and_log("Excluding unpaired tomograms from training")

            all_tomograms = list(tomo_coord_dict.keys())

            if unmatched_coords:
                print_and_log("Warning: The following coordinate files have no matching tomograms and will be ignored:")
                for coord in unmatched_coords:
                    print_and_log(f"  - {os.path.basename(coord)}")

            # Check preprocessing status and prompt user if necessary
            if prompt_user_for_preprocessing(all_tomograms, args.preprocess_dir, args.disable_topaz_preprocess):
                print_and_log("Unstacking tomograms and extending coordinates")
                all_slices, all_extended_coords = slice_and_write_tomograms(tomo_coord_dict, unstack_dir, args.slice_thickness, args.num_slices, mark_coords=False, max_workers=args.cpus)

                print_and_log("Saving all extended coordinates for training")
                if all_extended_coords:
                    write_extended_coordinates(extended_coords_file, all_extended_coords)
                else:
                    print_and_log("No coordinates to extend. Skipping coordinate file creation.")

                if args.disable_topaz_preprocess:
                    print_and_log("Performing simple normalization on all slices")
                    normalize_slices(all_slices, args.preprocess_dir, args.cpus)
                else:
                    print_and_log("Running Topaz preprocess on all slices")
                    run_topaz_preprocess(unstack_dir, args.preprocess_dir, args.scale, args.sample, args.preprocess_workers, args.device, args.niters)

                create_preprocessing_metadata(all_tomograms, args.preprocess_dir, args.disable_topaz_preprocess)

            print_and_log("Splitting the dataset into test and train")
            split_dataset(args.preprocess_dir, extended_coords_file, args.test_split)

            print_and_log("Running Topaz train")
            os.makedirs(f"{args.output}/model", exist_ok=True)
            model_save_prefix = f"{args.output}/model/{args.model_name}"
            test_train_curve = f"{args.output}/model/train_test_curve.txt"
            avg_num_slices = average_slices_per_tomogram(all_tomograms)
            expected_particles_per_slice = (args.expected_particles * (2 * args.num_slices + 1)) / (avg_num_slices // args.slice_thickness)
            run_topaz_train(args.preprocess_dir, f"{args.preprocess_dir}/particles_train.txt", f"{args.preprocess_dir}/image_list_test.txt",
                            f"{args.preprocess_dir}/particles_test.txt", model_save_prefix, test_train_curve, expected_particles_per_slice,
                            args.num_workers, device=args.device, format=args.format, image_ext=args.image_ext, k_fold=args.k_fold,
                            fold=args.fold, radius=args.radius, method=args.method, slack=args.slack, autoencoder=args.autoencoder, l2=args.l2,
                            learning_rate=args.learning_rate, natural=args.natural, minibatch_size=args.minibatch_size,
                            minibatch_balance=args.minibatch_balance, epoch_size=args.epoch_size, num_epochs=args.num_epochs,
                            no_pretrained=args.no_pretrained, model=args.model, units=args.units,
                            dropout=args.dropout, bn=args.bn, pooling=args.pooling, unit_scaling=args.unit_scaling, ngf=args.ngf,
                            save_prefix=args.save_prefix, test_batch_size=args.test_batch_size)
            plot_training_metrics(test_train_curve, train_sample_rate = args.plotting_train_sample_rate)
            _, _ = find_best_epoch(test_train_curve)

        elif args.mode == "extract":
            print_and_log("Extracting particles from all tomograms")

            # Check preprocessing status and prompt user if necessary
            if prompt_user_for_preprocessing(tomogram_files, args.preprocess_dir, args.disable_topaz_preprocess):
                print_and_log("Unstacking tomograms")
                # Create a dictionary with tomogram paths as both keys and values
                tomo_dict = {tomo: tomo for tomo in tomogram_files}
                all_slices = slice_and_write_tomograms(tomo_dict, unstack_dir, args.slice_thickness, max(args.num_slices), mark_coords=False, max_workers=args.cpus)
                
                if args.disable_topaz_preprocess:
                    print_and_log("Performing simple normalization on all slices")
                    normalize_slices(all_slices, args.preprocess_dir, args.cpus)
                else:
                    print_and_log("Running Topaz preprocess on all slices")
                    run_topaz_preprocess(unstack_dir, args.preprocess_dir, args.scale, args.sample, args.preprocess_workers, args.device, args.niters)

                create_preprocessing_metadata(tomogram_files, args.preprocess_dir, args.disable_topaz_preprocess)
            else:
                print_and_log("Using existing preprocessed slices")

            is_competitive = len(args.model_files) > 1
            extraction_type = "competitive " if is_competitive else ""
            print_and_log(f"Running {extraction_type}Topaz extract on all slices")
            os.makedirs(f"{args.output}/predicted", exist_ok=True)
            all_predictions = run_competitive_extraction(args.tomograms, args.model_files, args.particle_radii, args.num_slices, args.names, 
                                                         args.preprocess_dir, args.scale, threshold=args.threshold,
                                                         assignment_radius=args.assignment_radius, min_radius=args.min_radius, 
                                                         max_radius=args.max_radius, step_radius=args.step_radius, 
                                                         extract_workers=args.extract_workers, targets=args.targets, 
                                                         device=args.device, output=args.output, slice_thickness=args.slice_thickness, cpus=args.cpus)

            print_and_log("Normalizing scores")
            normalized_predictions = normalize_scores(all_predictions)

            picking_type = "Performing competitive picking" if is_competitive else "Selecting final particles"
            print_and_log(picking_type)
            final_particles = competitive_picking(normalized_predictions, args.particle_radii, max_workers=args.cpus)

            print_and_log("Saving aggregated coordinates")
            save_aggregated_coordinates(final_particles, args.threshold, f"{args.output}/predicted")

        end_time = time.time()
        time_str = time_diff(end_time - start_time)
        print_and_log(f"Topaz 2.5D {'training' if args.mode == 'train' else 'extracting' if args.mode == 'extract' else 'processing'} completed in \033[1m{time_str}\033[0m.")

    except Exception as e:
        print_and_log(f"Error occurred: {e}", level=logging.ERROR)
        sys.exit(1)

if __name__ == "__main__":
    main()
