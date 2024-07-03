#!/usr/bin/env python3
#
# Author: Alex J. Noble, assisted by GPT4o, 2024 @SEMC, MIT License
#
# Topaz 2.5D
#
# This script extends Topaz (2D) to pick slices of 3D tomograms given tomograms and
# corresponding 3D coordinates. It extends the capabilities of Topaz by preprocessing
# tomograms, training models on 2D slices, & aggregating 2D predictions into 3D coordinates.
#
# Dependencies: Topaz
#
# Topaz is distributed under the GPL-3.0 license. For details, see:
# - GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.en.html
# Topaz source code: https://github.com/tbepler/topaz/
# Ensure compliance with license terms when obtaining and using Topaz.
__version__ = "1.0.0"

import argparse
import logging
import inspect
import numpy as np
import mrcfile
import subprocess
import random
import os

global_verbosity = 1  # Default verbosity level

def print_and_log(message, level=logging.INFO):
    """
    Prints and logs a message with the specified level, including debug details for verbosity level 3.

    :param str message: The message to print and log.
    :param int level: The logging level for the message (e.g., logging.INFO, logging.DEBUG).

    If verbosity is set to 3, the function logs additional details about the caller,
    including module name, function name, line number, and function parameters.

    This function writes logging information to the disk.
    """
    logger = logging.getLogger()
    if global_verbosity < 3:
        logger.log(level, message)
    else:
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        line_no = caller_frame.f_lineno
        module_name = caller_frame.f_globals["__name__"]
        if func_name != 'print_and_log':
            args, _, _, values = inspect.getargvalues(caller_frame)
            args_info = ', '.join([f"{arg}={values[arg]}" for arg in args])
            detailed_message = f"{message} - Debug - Module: {module_name}, Function: {func_name}({args_info}), Line: {line_no}"
            logger.log(level, detailed_message)
        else:
            logger.log(level, message)

def setup_logging(output_dir):
    """
    Sets up logging configuration.

    :param str output_dir: The directory where the log file will be saved.
    """
    logging.basicConfig(filename=f"{output_dir}/topaz2_5d.log", level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """
    Parses command-line arguments.

    :return argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Topaz 2.5D: Particle picking 3D cryoET tomograms by training and predicting on tomogram slices.")
    parser.add_argument("mode", choices=["train", "extract"], help="Mode to run the script in: train or extract")
    parser.add_argument("-t", "--tomograms", nargs='+', required=True, help="Paths to the input tomograms (MRC files)")
    parser.add_argument("-c", "--coordinates", nargs='+', required=True, help="Paths to the input coordinates files")
    parser.add_argument("-o", "--output", required=True, help="Directory to save output coordinate files")
    parser.add_argument("-s", "--slice_thickness", type=int, default=1, help="Thickness of each slice for training/prediction where slices are averaged (default: 1)")
    parser.add_argument("-n", "--num_slices", type=int, required=True, help="Number of slices to extend symmetrically vertically for training")
    parser.add_argument("-e", "--expected_particles", type=int, required=True, help="Expected number of particles per tomogram slice for training")
    parser.add_argument("-r", "--particle_radius", type=int, required=True, help="Particle radius for extraction")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="Number of worker threads to use for topaz train (default: 8)")
    parser.add_argument("-p", "--preprocess_dir", required=True, help="Directory for Topaz preprocessing output")
    parser.add_argument("-T", "--test_split", type=float, default=0.2, help="Percentage of tomogram slices to use for testing; remaining are used for training (default: 0.2)")
    parser.add_argument("-V", "--verbosity", type=int, default=1, help="Verbosity level (default: 1)")
    parser.add_argument("-v", "--version", action="version", help="Show version number and exit", version=f"Topaz 2.5D v{__version__}")

    # Topaz preprocess specific arguments
    parser.add_argument("--scale", type=int, default=4, help="Rescaling factor for image downsampling (default: 4)")
    parser.add_argument("--preprocess_workers", type=int, default=0, help="Number of processes to use for parallel image downsampling (default: 0)")
    parser.add_argument("--pixel_sampling", type=int, default=100, help="Pixel sampling factor for model fit (default: 100)")
    parser.add_argument("--niters", type=int, default=200, help="Number of iterations to run for model fit (default: 200)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for model initialization (default: 1)")
    
    return parser.parse_args()

def readmrc(mrc_path):
    """
    Read an MRC file and return its data as a NumPy array.

    :param str mrc_path: The file path of the MRC file to read.
    :return numpy.ndarray: The data of the MRC file as a NumPy array.
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        numpy_array = np.array(data)
    return numpy_array

def writemrc(mrc_path, numpy_array, voxelsize=1.0):
    """
    Write a 3D NumPy array as an MRC file with specified voxel size.

    :param str mrc_path: The file path of the MRC file to write.
    :param numpy.ndarray numpy_array: The 3D NumPy array to be written.
    :param float voxelsize: The voxel size in Angstroms, assumed equal for all dimensions.
    :raises ValueError: If input numpy_array is not 3D.
    """
    if numpy_array.ndim not in [3]:
        raise ValueError("Input array must be 3D")
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(numpy_array)
        mrc.voxel_size = (voxelsize, voxelsize, voxelsize)
        mrc.update_header_from_data()
        mrc.update_header_stats()

def slice_tomograms(tomogram, slice_thickness=1):
    """
    Slice a 3D tomogram into 2D slices by averaging over the specified thickness.

    :param numpy.ndarray tomogram: The 3D tomogram as a NumPy array.
    :param int slice_thickness: The thickness of each slice to average over.
    :return list: A list of 2D slices.
    """
    slices = []
    for i in range(0, tomogram.shape[2], slice_thickness):
        if i + slice_thickness <= tomogram.shape[2]:
            # Average over the slice_thickness to create a single 2D slice
            slice_2d = np.mean(tomogram[:, :, i:i+slice_thickness], axis=2)
            slices.append(slice_2d)
    return slices

def extend_coordinates(coords, num_slices):
    """
    Extend coordinates upward and downward by a specified number of slices.

    :param list coords: List of (x, y, z) coordinates.
    :param int num_slices: Number of slices to extend.
    :return list: Extended list of coordinates.
    """
    extended_coords = []
    for x, y, z in coords:
        for i in range(-num_slices, num_slices + 1):
            extended_coords.append((x, y, z + i))
    return extended_coords

def run_topaz_preprocess(input_dir, output_dir, scale, num_workers, pixel_sampling, niters, seed):
    """
    Run topaz preprocess command with additional options.

    :param str input_dir: Directory of input files.
    :param str output_dir: Directory to save output files.
    :param int scale: Rescaling factor for image downsampling.
    :param int num_workers: Number of processes to use for parallel image downsampling.
    :param int pixel_sampling: Pixel sampling factor for model fit.
    :param int niters: Number of iterations to run for model fit.
    :param int seed: Random seed for model initialization.
    """
    command = [
        "topaz", "preprocess", "-v",
        "-s", str(scale),
        "-t", str(num_workers),
        "--pixel-sampling", str(pixel_sampling),
        "--niters", str(niters),
        "--seed", str(seed),
        "-o", output_dir,
        input_dir + "/*.mrc"
    ]
    subprocess.run(command, check=True)

def run_topaz_train_test_split(image_dir, particles_file, test_split, seed=None):
    """
    Run topaz train_test_split command based on the percentage of data for testing,
    with random selection of images.

    :param str image_dir: Directory of images.
    :param str particles_file: File with particle coordinates.
    :param float test_split: Percentage of data to use for testing.
    :param int seed: Random seed for reproducibility. If None, a random seed will be generated.
    """
    # Read the number of images in the directory
    num_images = len([f for f in os.listdir(image_dir) if f.endswith('.mrc')])
    num_test_images = max(1, int(num_images * test_split))
    
    # If no seed is provided, generate a random one
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    command = [
        "topaz", "train_test_split",
        "-n", str(num_test_images),
        "--image-dir", image_dir,
        "--seed", str(seed),
        particles_file
    ]
    
    print(f"Running train_test_split with seed: {seed}")
    subprocess.run(command, check=True)

def run_topaz_train(train_images_dir, train_targets_file, model_save_prefix, expected_particles, num_workers=8):
    """
    Run topaz train command.

    :param str train_images_dir: Directory of training images.
    :param str train_targets_file: File with training targets.
    :param str model_save_prefix: Prefix for saving the trained model.
    :param int expected_particles: Expected number of particles per image.
    :param int num_workers: Number of worker threads to use.
    """
    command = [
        "topaz", "train",
        "-n", str(expected_particles),
        "--num-workers", str(num_workers),
        "--train-images", train_images_dir,
        "--train-targets", train_targets_file,
        "--save-prefix", model_save_prefix
    ]
    subprocess.run(command, check=True)

def run_topaz_extract(model_file, input_dir, output_file, particle_radius, scale_factor=1):
    """
    Run topaz extract command.

    :param str model_file: Path to the trained model file.
    :param str input_dir: Directory of input files.
    :param str output_file: Path to save the output coordinates.
    :param int particle_radius: Radius of particles to extract.
    :param int scale_factor: Scaling factor for the coordinates.
    """
    command = [
        "topaz", "extract",
        "-r", str(particle_radius),
        "-x", str(scale_factor),
        "-m", model_file,
        "-o", output_file,
        input_dir + "/*.mrc"
    ]
    subprocess.run(command, check=True)

def aggregate_predictions(predictions, num_slices, slice_thickness=1, tolerance=2, slice_window=10):
    """
    Aggregate 2D predictions into 3D coordinates.

    :param list predictions: List of (x, y, z) predictions.
    :param int num_slices: Number of slices to consider.
    :param int slice_thickness: Thickness of each slice used in prediction.
    :param int tolerance: Tolerance for coordinate similarity.
    :param int slice_window: Number of slices to group together.
    :return list: Aggregated 3D coordinates.
    """
    aggregated_coords = []
    z_step = slice_thickness  # Adjust for slice thickness

    for i in range(0, len(predictions) - num_slices, num_slices):
        slice_group = predictions[i:i+num_slices]
        avg_x = sum([p[0] for p in slice_group]) / num_slices
        avg_y = sum([p[1] for p in slice_group]) / num_slices
        avg_z = (sum([p[2] for p in slice_group]) / num_slices) * z_step

        if len(set((round(p[0]), round(p[1])) for p in slice_group)) <= tolerance:
            aggregated_coords.append((avg_x, avg_y, avg_z))
    
    return aggregated_coords

def read_coordinates(coord_path):
    """
    Read coordinates from a file and return as a list of tuples.

    :param str coord_path: Path to the coordinate file.
    :return list: List of (x, y, z) coordinates.
    """
    coordinates = []
    with open(coord_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            coordinates.append((x, y, z))
    return coordinates

def write_coordinates(coord_path, coordinates):
    """
    Write coordinates to a file.

    :param str coord_path: Path to the coordinate file.
    :param list coordinates: List of (x, y, z) coordinates.
    """
    with open(coord_path, 'w') as file:
        for coord in coordinates:
            file.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

def read_predictions(pred_path):
    """
    Read predictions from a file and return as a list of tuples.

    :param str pred_path: Path to the predictions file.
    :return list: List of (x, y, z) predictions.
    """
    predictions = []
    with open(pred_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.strip().split())
            predictions.append((x, y, z))
    return predictions

def main():
    """
    Main function: Determines mode, loops through processes, prints and logs each step.
    """
    args = parse_args()
    global global_verbosity
    global_verbosity = args.verbosity
    
    setup_logging(args.output)
    
    try:
        if args.mode == "train":
            print_and_log("Training on all tomograms")
            
            all_slices = []
            all_extended_coords = []
            
            for tomogram_path, coord_path in zip(args.tomograms, args.coordinates):
                print_and_log(f"Processing tomogram: {tomogram_path} with coordinates: {coord_path}")
                
                print_and_log("Reading tomogram")
                tomogram = readmrc(tomogram_path)
                
                print_and_log("Slicing tomogram")
                slices = slice_tomograms(tomogram, args.slice_thickness)
                all_slices.extend(slices)
                
                print_and_log("Reading and extending coordinates")
                coordinates = read_coordinates(coord_path)
                extended_coords = extend_coordinates(coordinates, args.num_slices)
                all_extended_coords.extend(extended_coords)
            
            print_and_log("Running Topaz preprocess on all slices")
            preprocessed_dir = f"{args.preprocess_dir}/all_tomograms"
            os.makedirs(preprocessed_dir, exist_ok=True)
            for i, slice_2d in enumerate(all_slices):
                writemrc(f"{preprocessed_dir}/slice_{i:04d}.mrc", slice_2d)
            
            print_and_log("Running Topaz preprocess")
            run_topaz_preprocess(preprocessed_dir, preprocessed_dir, args.scale, args.preprocess_workers, args.pixel_sampling, args.niters, args.seed)
            
            print_and_log("Saving all extended coordinates for training")
            extended_coords_file = f"{preprocessed_dir}/extended_coordinates.txt"
            write_coordinates(extended_coords_file, all_extended_coords)
            
            print_and_log("Running Topaz train_test_split")
            run_topaz_train_test_split(preprocessed_dir, extended_coords_file, args.test_split, args.seed)
            
            print_and_log("Running Topaz train")
            model_save_prefix = f"{args.output}/model/all_tomograms"
            run_topaz_train(preprocessed_dir, f"{preprocessed_dir}/particles_train.txt", model_save_prefix, args.expected_particles, args.num_workers)
        
        elif args.mode == "extract":
            print_and_log("Extracting particles from all tomograms")
            
            for tomogram_path in args.tomograms:
                print_and_log(f"Processing tomogram: {tomogram_path}")
                
                print_and_log("Reading tomogram")
                tomogram = readmrc(tomogram_path)
                
                print_and_log("Slicing tomogram")
                slices = slice_tomograms(tomogram, args.slice_thickness)
                
                preprocessed_dir = f"{args.preprocess_dir}/{os.path.basename(tomogram_path).split('.')[0]}"
                os.makedirs(preprocessed_dir, exist_ok=True)
                for i, slice_2d in enumerate(slices):
                    writemrc(f"{preprocessed_dir}/slice_{i:04d}.mrc", slice_2d)
                
                print_and_log("Running Topaz preprocess")
                run_topaz_preprocess(preprocessed_dir, preprocessed_dir, args.scale, args.preprocess_workers, args.pixel_sampling, args.niters, args.seed)
                
                print_and_log("Running Topaz extract")
                predictions_file = f"{args.output}/predicted/{os.path.basename(tomogram_path).split('.')[0]}_predicted_coordinates.txt"
                model_save_prefix = f"{args.output}/model/all_tomograms"
                run_topaz_extract(f"{model_save_prefix}_epoch10.sav", preprocessed_dir, predictions_file, args.particle_radius, scale_factor=1)
                
                print_and_log("Aggregating predictions")
                predictions = read_predictions(predictions_file)
                aggregated_coords = aggregate_predictions(predictions, args.num_slices, slice_thickness=args.slice_thickness)
                
                print_and_log("Saving aggregated coordinates")
                aggregated_coords_file = f"{args.output}/aggregated/{os.path.basename(tomogram_path).split('.')[0]}_aggregated_coordinates.txt"
                write_coordinates(aggregated_coords_file, aggregated_coords)
        
        print_and_log("Topaz 2.5D processing complete.")
    
    except Exception as e:
        print_and_log(f"Error occurred: {e}", level=logging.ERROR)
        raise

if __name__ == "__main__":
    main()
