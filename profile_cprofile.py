# Profiling Step 4 of Part 2 of Project
# by Maheen Khan
# new profiling script using cProfile

import cProfile
import pstats
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def profile_model_script():
    # Set up the relative path to the model script
    script_dir = os.path.dirname(__file__)
    model_script = os.path.join(script_dir, 'src', 'models', 'ABBV_StockPrediction1.py')

    # Ensure the script exists
    if not os.path.exists(model_script):
        print(f"Model script not found: {model_script}")
        sys.exit(1)

    # Define the path for saving profiling results
    profile_output = os.path.join(script_dir, 'cprofile_results.txt')

    # Run the model script with cProfile
    profile = cProfile.Profile()
    profile.enable()

    # Read and execute the model script
    with open(model_script, 'rb') as f:
        exec(compile(f.read(), model_script, 'exec'), globals(), locals())

    profile.disable()

    # Save the profiling results to a file
    with open(profile_output, 'w') as f:
        ps = pstats.Stats(profile, stream=f)
        ps.strip_dirs().sort_stats('time').print_stats()

    print(f"cProfile results saved to {profile_output}")

if __name__ == "__main__":
    profile_model_script()
