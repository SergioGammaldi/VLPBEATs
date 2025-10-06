# VLPBEATs
VLPBEATs

VLPBEATs is a Python framework for detecting Very-Long-Period (VLP) signals in seismic data. It enables near real-time monitoring of volcanic activity by automatically identifying VLP events.

Installation

The recommended way to install VLPBEATs is via Conda using the provided environment file.

Clone the repository (if not already done):

git clone https://github.com/your-username/VLPBEATs.git
cd VLPBEATs


Create and activate the Conda environment:

conda env create -f vlpbeats.yml
conda activate vlpbeats


This will install all required dependencies for VLPBEATs.

Setup

Configure the input file input_vlpbeats.yml according to your VLP dataset. You can follow the example configuration provided by:

Gammaldi et al., 2025 (submitted to Scientific Reports):
A Near Real-Time Framework for Detecting Very-Long-Period Signals: New Perspectives for Volcano Monitoring

Adjust the parameters in the YAML file according to your station names, data paths, and desired VLP events.

Usage

Once the environment is activated and the YAML input file is configured, you can run the framework using:

python3 main.py --config input_vlpbeats.yml

for any problems contact sergio.gammaldi@ingv.it