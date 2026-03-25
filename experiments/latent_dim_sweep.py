import os
import yaml
import subprocess
import pandas as pd

def run_sweep():
    # Define the sweep range: 2, 4, 8, 16, 32, 64, 128
    latent_dims = [2**i for i in range(1, 8)]
    base_config_path = "configs/vae_mnist_2layer.yaml"
    results_file = "experiments/sweep_results.csv"
    
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    sweep_data = []

    for dim in latent_dims:
        print(f"\n--- Running Experiment: Latent Dims = {dim} ---")
        
        # 1. Update config for this run
        base_config['model']['latent_dims'] = dim
        
        # Create a unique experiment name
        exp_name = f"mnist_2layer_z{dim}"
        temp_config_path = f"configs/tmp_sweep_{dim}.yaml"
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(base_config, f)

        # 2. Execute training
        # We call the existing train.py as a subprocess
        cmd = ["python", "train.py", temp_config_path]
        subprocess.run(cmd)

        # 3. Collect results
        # Assuming the train.py saves a checkpoint or log we can read
        # For now, let's assume we extract the final loss from the output directory
        out_dir = f"outputs/tmp_sweep_{dim}"
        # (Logic to read final loss from a saved file if available)
        
        # Cleanup temp config
        os.remove(temp_config_path)

    print("\nSweep Complete.")

if __name__ == "__main__":
    run_sweep()