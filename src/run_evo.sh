#!/bin/bash

# Define the criteria to test
criteria=("sparseness" "average_sensitivity") # "faithfulness"
multi_opt_criteria=("sparseness,faithfulness" "average_sensitivity,faithfulness") # "sparseness,average_sensitivity" 
rules=("gamma") # "epsilon" "alpha_beta" 
# Create a directory for the outputs if it doesn't exist
mkdir -p outputs

# Loop through each criterion and run the command
for criterion in "${criteria[@]}"; do
    for rule in "${rules[@]}"; do
        echo "Running with criterion: $criterion and rule: $rule" 
        
        # Define the output file name
        output_file="outputs/output_${criterion}_${rule}.txt"
        
        # Run the command and redirect output to the file
        if [ "$rule" == "alpha_beta" ]; then
            python main.py --dataset imagenet --batch_size 64 --init_value 2 --generations 100 --rule "$rule" --multi_opt --multi_opt_criterion "$criterion" > "$output_file" 2>&1
        elif [ "$rule" == "epsilon" ]; then
            python main.py --dataset imagenet --batch_size 64 --init_value 0.25 --generations 100 --rule "$rule" --multi_opt --multi_opt_criterion "$criterion" > "$output_file" 2>&1
        elif [ "$rule" == "gamma" ]; then
            python main.py --dataset imagenet --batch_size 128 --init_value 1 --generations 100 --rule "$rule" --criterion "$criterion" > "$output_file" 2>&1
        fi
        
        echo "Completed $criterion. Output saved to $output_file"
        echo "----------------------------------------"
    done
done

echo "All runs completed!"