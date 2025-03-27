#!/usr/bin/env python3

import os
import subprocess
import shutil
import time
from datetime import datetime

# Define all modes and their corresponding rule names
mode_rule_combinations = [
    # ("uniform_rule", ["alpha_beta", "gamma", "epsilon"]),
    ("uniform_value", [None]),  # uniform_value doesn't need a rule_name
    # ("layer_type", [None]),     # layer_type doesn't need a rule_name
    # ("composite_layer", [None]) # composite_layer doesn't need a rule_name
]

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create a directory to store all CMA-ES output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Run each combination
for mode, rule_names in mode_rule_combinations:
    for rule_name in rule_names:
        # Create command
        if rule_name:
            cmd = f"python main.py --dataset imagenet --batch_size 16 --generations 200 --rule {rule_name}"
            log_file = f"logs/EvoLRP_{mode}_{rule_name.upper()}.txt"
        else:
            cmd = f"python main.py --dataset imagenet --batch_size 16 --generations 200"
            log_file = f"logs/EvoLRP_{mode}.txt"
        
        # Print current run
        print(f"Running: {cmd}")
        print(f"Logging to: {log_file}")
        
        # Execute command and log output
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Stream output to both file and console
            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end='')
                
            process.wait()
        
        # Create a subdirectory for this run's results
        if rule_name:
            run_dir = f"{results_dir}/{mode}_{rule_name}"
        else:
            run_dir = f"{results_dir}/{mode}"
        os.makedirs(run_dir, exist_ok=True)
        
        # Move CMA-ES output files to the run directory
        if os.path.exists("outcmaes"):
            for file in os.listdir("outcmaes"):
                src = os.path.join("outcmaes", file)
                dst = os.path.join(run_dir, file)
                shutil.copy2(src, dst)
        
        print(f"Completed run for {mode}" + (f" with {rule_name}" if rule_name else ""))
        print(f"Results saved to {run_dir}")
        print("-" * 80)
        
        # Small delay to ensure files are properly closed
        time.sleep(1)

print(f"All runs completed. Results saved in {results_dir}")