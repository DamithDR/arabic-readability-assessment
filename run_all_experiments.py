# -*- coding: utf-8 -*-
"""
Run all DARES experiments
DARES1.0: 3 settings x 5 models x 2 types = 30 experiments
DARES2.0: 4 settings x 5 models x 2 types = 40 experiments
Total: 70 experiments
"""
import subprocess
import sys
import time
from datetime import datetime

# Models to test
MODELS = [
    ("CAMeL-Lab/bert-base-arabic-camelbert-mix", "bert", "CAMeLBERTmix"),
    ("aubmindlab/bert-base-arabertv2", "bert", "AraBERTv2"),
    ("aubmindlab/araelectra-base-discriminator", "electra", "AraELECTRA"),
    ("google-bert/bert-base-multilingual-cased", "bert", "mBERT"),
    ("xlm-roberta-base", "xlmroberta", "XLM-R"),
]

# DARES1.0 settings (3 settings)
DARES1_SETTINGS = [
    # Fine-grained
    ("raw", None, "Fine"),
    ("append_word", "Word", "Fine"),
    ("append_filename", "Arabic_Filename", "Fine"),
    # Coarse-grained
    ("raw_cat", None, "Coarse"),
    ("append_word_categorised", "Word", "Coarse"),
    ("append_filename_categorised", "Arabic_Filename", "Coarse"),
]

# DARES2.0 settings (4 settings - includes word_file)
DARES2_SETTINGS = [
    # Fine-grained
    ("raw", None, "Fine"),
    ("append_word", "Word", "Fine"),
    ("append_filename", "Arabic_Filename", "Fine"),
    ("word_file", "word_file", "Fine"),
    # Coarse-grained
    ("raw_cat", None, "Coarse"),
    ("append_word_categorised", "Word", "Coarse"),
    ("append_filename_categorised", "Arabic_Filename", "Coarse"),
    ("word_file_cat", "word_file_cat", "Coarse"),
]

# Training parameters
NUM_EPOCHS = 4
N_FOLD = 3
CUDA_DEVICE = 0
LR = 0.00001

def run_experiment(dataset, experiment_script, model_name, model_type, model_short, run_mode, append_column, grain_type):
    """Run a single experiment"""
    cmd = [
        sys.executable, "-m", experiment_script,
        "--model_name", model_name,
        "--model_type", model_type,
        "--num_train_epochs", str(NUM_EPOCHS),
        "--run_mode", run_mode,
        "--n_fold", str(N_FOLD),
        "--cuda_device", str(CUDA_DEVICE),
        "--lr", str(LR),
        "--save_predictions", "True",
    ]
    
    if append_column:
        cmd.extend(["--append_column", append_column])
    
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {dataset} | {model_short} | {run_mode} | {grain_type}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Completed in {elapsed/60:.1f} minutes")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[FAILED] Error after {elapsed/60:.1f} minutes: {e}")
        return False, elapsed
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User cancelled")
        return False, 0

def main():
    print("="*70)
    print("DARES Experiments Runner")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODELS)}")
    print(f"DARES1.0 experiments: {len(DARES1_SETTINGS) * len(MODELS)}")
    print(f"DARES2.0 experiments: {len(DARES2_SETTINGS) * len(MODELS)}")
    print(f"Total experiments: {(len(DARES1_SETTINGS) + len(DARES2_SETTINGS)) * len(MODELS)}")
    print()
    
    results = []
    total_start = time.time()
    
    # Run DARES1.0 first
    print("\n" + "="*70)
    print("PHASE 1: DARES1.0 Experiments")
    print("="*70)
    
    for model_name, model_type, model_short in MODELS:
        for run_mode, append_column, grain_type in DARES1_SETTINGS:
            success, elapsed = run_experiment(
                "DARES1.0",
                "experiments.dares1.0_assess",
                model_name, model_type, model_short,
                run_mode, append_column, grain_type
            )
            results.append({
                "dataset": "DARES1.0",
                "model": model_short,
                "run_mode": run_mode,
                "grain": grain_type,
                "success": success,
                "time_min": elapsed/60
            })
    
    # Run DARES2.0
    print("\n" + "="*70)
    print("PHASE 2: DARES2.0 Experiments")
    print("="*70)
    
    for model_name, model_type, model_short in MODELS:
        for run_mode, append_column, grain_type in DARES2_SETTINGS:
            success, elapsed = run_experiment(
                "DARES2.0",
                "experiments.dares2.0_assess",
                model_name, model_type, model_short,
                run_mode, append_column, grain_type
            )
            results.append({
                "dataset": "DARES2.0",
                "model": model_short,
                "run_mode": run_mode,
                "grain": grain_type,
                "success": success,
                "time_min": elapsed/60
            })
    
    # Summary
    total_time = time.time() - total_start
    successful = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['dataset']} | {r['model']} | {r['run_mode']}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
