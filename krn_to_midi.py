import os
from music21 import converter

def convert_krn_to_midi(input_dir, output_dir):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of .krn files in the input directory
    krn_files = [f for f in os.listdir(input_dir) if f.endswith('.krn')]

    # Process each .krn file
    for krn_file in krn_files:
        krn_file_path = os.path.join(input_dir, krn_file)
        # Load the .krn file with music21
        score = converter.parse(krn_file_path)
        score.write('midi', fp=os.path.join(output_dir, krn_file.replace('.krn', '.midi')))

convert_krn_to_midi("deutschl/erk","melodies/")
