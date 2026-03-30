import argparse

from recon_datavis.hdf5_visualizer import HDF5Visualizer
from recon_datavis.utils import get_files_ending_with


parser = argparse.ArgumentParser()
parser.add_argument('-folders', nargs='+', help='list of folders containing hdf5s')
parser.add_argument('-horizon', type=int, default=8)
args = parser.parse_args()

hdf5_fnames = get_files_ending_with(args.folders, '.hdf5')
hdf5_visualizer = HDF5Visualizer(hdf5_fnames)
hdf5_visualizer.run()
