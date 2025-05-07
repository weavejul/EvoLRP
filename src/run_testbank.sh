#!/bin/bash

# Create output directory for logs if it doesn't exist
mkdir -p logs

# Get current timestamp for the combined log
timestamp=$(date +"%Y%m%d_%H%M%S")
combined_log="logs/combined_results_${timestamp}.txt"

# Array of XAI methods to test
methods=("lrp") # "Lime" "LayerGradCam" "IntegratedGradients" "lrp" 

# Array of criteria to test
criteria=("sparseness" "average_sensitivity" "faithfulness") # "sparseness" "average_sensitivity"
multi_criteria=("sparseness,faithfulness" "average_sensitivity,faithfulness" "sparseness,average_sensitivity")

# rule type
rule_type=( "alpha_beta")

# Base command
base_cmd="python testbank.py -i /home/neuro/XAIProject/data/ImageNet"

# Write header to combined log
echo "XAI Method Comparison Results - $(date)" | tee "$combined_log"
echo "=====================================" | tee -a "$combined_log"

# Loop through each method
for method in "${methods[@]}"; do
    for criterion in "${multi_criteria[@]}"; do
        for rule in "${rule_type[@]}"; do
            echo "Running testbank with $method and $criterion..." | tee -a "$combined_log"
                echo "================================" | tee -a "$combined_log"
                
                # Create method-specific log file
                method_log="logs/${method}_${criterion}_${timestamp}.txt"
                
                # Split the criterion by comma and iterate
                IFS=',' read -ra CRITERIA <<< "$criterion"
                for criteria in "${CRITERIA[@]}"; do
                    # Run command and capture output to both method-specific and combined logs
                    if [ "$method" == "Lime" ]; then
                        $base_cmd --xai-method "$method" --criterion "$criterion" -b 1 2>&1 | tee "$method_log" | tee -a "$combined_log"
                    elif [ "$method" == "lrp" ]; then
                        $base_cmd --xai-method "$method" --criterion "$criteria" -r "$rule" -b 32 --lrp-params saved_params/opt_${rule}_${criteria}.json 2>&1 | tee "$method_log" | tee -a "$combined_log"
                    else
                    $base_cmd --xai-method "$method" --criterion "$criterion" 2>&1 | tee "$method_log" | tee -a "$combined_log"
                    fi
                done
            echo -e "\n" | tee -a "$combined_log"
        done
    done
done

echo "All tests completed!" | tee -a "$combined_log"
echo "Results have been saved to individual files in the logs directory"
echo "Combined results are available in: $combined_log"