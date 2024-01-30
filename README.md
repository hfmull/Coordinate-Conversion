# Coordinate-Conversion

This is a simple python script for converting between [XYZ](https://en.wikipedia.org/wiki/XYZ_file_format) and [ZMAT](https://en.wikipedia.org/wiki/Z-matrix_(chemistry)) formats for molecule coordinates. 

## Installation

To use the script, simply `git clone` either this repository or your own fork. The only dependencies needed are NumPy and QCElemental, both of which can be easily installed through Pip or Conda.

## Usage

The script expects 2 positional arguments and can accept 2 options. The most basic usage is `python path/to/script/coordinate_conversion.py input_filename input_format` where `input_filename` is the path to the file you wish to convert and `input_format` is the starting format of the input file (i.e. `xyz` or `zmat`). With default options, the new coordinate string is output to the standard output, but this behavior can be modified with the `-o` flag. Providing a filename writes the output to that file, however using the `-o` flag with no additional input will write the output to the file `output`.

### Converting to ZMAT

When converting XYZ files to ZMAT, the input files must be formated as a standard XYZ file with the first 2 lines being the number of atoms and a comment line. The comment line will be used as the title line for the resulting ZMAT output. Cartesian coordinates can be provided in either angstrom or bohr, with the ZMAT output giving distances in the same units. However, if your geometry is described in bohr, use the `--bohr` option. The ZMAT indices are decided based on bond connectivity, and the function used requires the geometry in bohr, so without this option set you will get incorrect results. The order of the ZMAT is the same as the order of the XYZ input, so if this results in not having full bond connectivity for the 4 atom indices on a given line of the ZMAT, it will instead find the nearest atom and give a warning to the user.

### Converting to XYZ

Converting ZMAT files to XYZ files is much simpler. The comment line of the XYZ file will be taken from the title line of the ZMAT and the atom order will be the same. Units for the cartesian coordinates will be the same as the distance units used for the ZMAT values. 
