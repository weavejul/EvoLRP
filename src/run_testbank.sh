#!/bin/bash

# Create output directory for logs if it doesn't exist
mkdir -p logs

# Get current timestamp for the combined log
timestamp=$(date +"%Y%m%d_%H%M%S")
combined_log="logs/combined_results_${timestamp}.txt"

# Array of XAI methods to test
methods=("Lime" "LayerGradCam" "IntegratedGradients") # "Lime" "LayerGradCam" "IntegratedGradients" "lrp" 

# Array of criteria to test
criteria=("pixel_flipping" "faithfulness") # "sparseness" "average_sensitivity"

# rule type
rule_type=("alpha_beta" "epsilon" "gamma")

# Base command
base_cmd="python testbank.py -i /home/neuro/XAIProject/data/ImageNet"

# Write header to combined log
echo "XAI Method Comparison Results - $(date)" | tee "$combined_log"
echo "=====================================" | tee -a "$combined_log"

# Loop through each method
for method in "${methods[@]}"; do
    for criterion in "${criteria[@]}"; do
        echo "Running testbank with $method and $criterion..." | tee -a "$combined_log"
        echo "================================" | tee -a "$combined_log"
        
        # Create method-specific log file
        method_log="logs/${method}_${timestamp}.txt"
        
        # Run command and capture output to both method-specific and combined logs
        if [ "$method" == "Lime" ]; then
            $base_cmd --xai-method "$method" --criterion "$criterion" -b 1 2>&1 | tee "$method_log" | tee -a "$combined_log"
        elif [ "$method" == "lrp" ]; then
            $base_cmd --xai-method "$method" --criterion "$criterion" --lrp-params saved_params/opt_alpha_avgsens.json -r alpha_beta 2>&1 | tee "$method_log" | tee -a "$combined_log"
        else
            $base_cmd --xai-method "$method" --criterion "$criterion" 2>&1 | tee "$method_log" | tee -a "$combined_log"
        fi
        
        echo -e "\n" | tee -a "$combined_log"
    done
done

echo "All tests completed!" | tee -a "$combined_log"
echo "Results have been saved to individual files in the logs directory"
echo "Combined results are available in: $combined_log"