# Profiling Step 4 of Part 2 of Project
# by Maheen Khan
# new profiling script using cProfile

import cProfile
import pstats
import os
import sys
import pandas as pd

print("Current working directory:", os.getcwd())

# Set up the relative path to the stock prediction script
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, 'src', 'data', 'stock_data')
csv_file = 'ABBV.csv'
csv_path = os.path.join(data_dir, csv_file)

# Define the path for saving profiling results
profile_output = os.path.join(script_dir, 'cprofile_results.txt')

def main():
    # Reading the CSV file
    df = pd.read_csv(csv_path)

    # Running the stock prediction script with cProfile
    profile = cProfile.Profile()
    profile.enable()
    # Call your prediction function here passing df as an argument
    profile.disable()

    # Saving the profiling results to a file
    with open(profile_output, 'w') as f:
        ps = pstats.Stats(profile, stream=f)
        ps.strip_dirs().sort_stats('time').print_stats()

    print("cProfile results saved to {}".format(profile_output))

if __name__ == "__main__":
    main()
