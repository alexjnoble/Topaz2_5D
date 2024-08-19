# Topaz 2.5D
This script extends Topaz (2D) to pick slices of 3D tomograms given tomograms and corresponding 3D coordinates. It extends the capabilities of Topaz by preprocessing tomograms, training models on 2D slices, & aggregating 2D predictions into 3D coordinates.
Topaz 2.5D extends the capabilities of Topaz to pick particles in 3D cryoET tomograms by training and predicting on tomogram slices. It preprocesses tomograms, trains models on 2D slices, and aggregates 2D predictions into 3D coordinates.

#### Features

- Generates 2D slices from 3D tomograms for training and prediction.
- Trains Topaz models on 2D slices with extended 3D coordinates.
- Aggregates 2D predictions into 3D coordinates, removing duplicates.
- Training metrics are plotted and the best iteration is determined in 2 different ways.
- Topaz scores from 2D predictions are retained in the 3D coordinates.
- Topaz general model works; training is not necessary.
- Competitive picking supported; pick multiple particle types simultaneously.
- Parallel processing for faster tomogram slicing and coordinate aggregation.
- Extensive customization options for preprocessing, training, and extraction.

## Installation

Topaz 2.5D requires Python 3, Topaz, and several dependencies, which can be installed using pip:

```bash
pip install matplotlib mrcfile numpy opencv-python pandas pillow scikit-learn scipy
```

Ensure Topaz is installed according to its [installation instructions](https://github.com/tbepler/topaz/tree/master?tab=readme-ov-file#installation).

To use Topaz 2.5D, download the topaz2_5d.py file directly and place it in your working directory or environment for use. Make it executable on Linux with this command: `chmod +x topaz2_5d.py`.

## Usage

Tomograms should be binned (downsampled betwee 4 and 16 times) prior to using Topaz 2.5D. Tomograms should be in .mrc format and coordinates should be in x,y,z format (space-delimited, no commas; one particle per line, one file per tomogram with the same basename as the tmogram).

The script can be run from the command line and takes a number of arguments.

```bash
./topaz2_5d.py train -t tomos/*mrc -c coords/*coords -o output -n 8 -e 1000 -V 2
```

Trains a model on `-t` tomograms matching tomos/*mrc using `-c` coordinates matching coords/*coords, with an output directory `-o` output, `-n` 8 slices (4 above and below each particle, which should be less than the minimum particle radius), `-e` 1000 expected particles per tomogram, and `-V` verbosity set to 2.

```bash
./topaz2_5d.py extract -t tomos/*mrc -m output/model/all_tomograms_epoch8.sav -o output -n 10 -r 12
```

Extracts particles from `-t` tomograms matching tomos/*mrc using a trained model `-m` output/model/all_tomograms_epoch8.sav, with an output directory `-o` output, `-n` 10 slices (5 above and below each particle, defining the particle size), and particle radius `-r` 12 (removes particles closer than 2 x 12 pixels from each other).

```bash
./topaz2_5d.py extract -t tomos/*mrc --names proteasome pretrained -m output/model/proteasome_epoch8.sav resnet16_u64 -n 8 20 -r 10 22 -V 2
```

Extracts particles from `-t` tomograms matching tomos/*mrc competitively for particle types with `--names` proteasome and pretrained using a trained model `-m` output/model/proteasome_epoch8.sav and a pretrained model resnet16_u64, `-n` 8 slices for proteasome and 20 slices for pretrained (ie. looking for objects twice the size of proteasomes), particle radius `-r` of 10 and 22 pixels for proteasome and pretrained, respectively, and `-V` verbosity set to 2.

Note: If you do not include a model for extraction, then a Topaz 2D general model will be used.

## Arguments

- `mode`: Choose between "train" or "extract" mode.
- `-t`, `--tomograms`: Paths to input tomograms (MRC files).
- `-c`, `--coordinates`: Paths to input coordinate files (for training mode).
- `-o`, `--output`: Directory to save output files.
- `-n`, `--num_slices`: Number of slices to extend symmetrically vertically.
- `-e`, `--expected_particles`: Expected number of particles per tomogram (for training).
- `-m`, `--model_file`: Trained model file path for extraction.
- `-r`, `--particle_radius`: Particle radius for extraction.

Additional arguments exist for fine-tuning the preprocessing, training, and extraction processes.

## Issues and Support

If you encounter any problems or have any questions about the script, please [Submit an Issue](https://github.com/alexjnoble/Topaz2_5D/issues).

## Contributions

Contributions are welcome! Please open a [Pull Request](https://github.com/alexjnoble/Topaz2_5D/pulls) or [Issue](https://github.com/alexjnoble/Topaz2_5D/issues).

## Author

This script was written by Alex J. Noble with assistance from Anthropic's Claude 3.5 Sonnet and OpenAI's GPT-4o models, 2024 at SEMC.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
