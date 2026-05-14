import os
import json
import shutil
from pathlib import Path

def clean_duplicates(root_dir):
    root_path = Path(root_dir)
    
    # Suffix mappings: suffix -> normalized_suffix
    normalization_map = {
        "_retrieval:knn_ctx:1024": "_retrieval:knn_ctx:1024",
        "_retrieval:knn_l2_ctx:1024": "_retrieval:knn_ctx:1024",
        "_sel:shap_100_retrieval:aware_imb_ctx:1024": "_retrieval:aware_imb_ctx:1024",
        "_retrieval:aware_imb_ctx:1024": "_retrieval:aware_imb_ctx:1024",
    }
    
    # Sort suffixes by length descending to match longest possible suffix first
    all_suffixes = sorted(normalization_map.keys(), key=len, reverse=True)

    if not root_path.exists():
        print(f"Error: {root_path} does not exist.")
        return

    for dataset_dir in sorted(root_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        print(f"Processing dataset: {dataset_dir.name}")
        
        # Group runs by (prefix, normalized_suffix)
        groups = {} # (prefix, norm_suffix) -> [path1, path2, ...]
        
        for run_dir in dataset_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            run_name = run_dir.name
            matched_suffix = None
            for s in all_suffixes:
                if run_name.endswith(s):
                    matched_suffix = s
                    break
            
            if matched_suffix:
                prefix = run_name[:-len(matched_suffix)]
                norm_suffix = normalization_map[matched_suffix]
                key = (prefix, norm_suffix)
                if key not in groups:
                    groups[key] = []
                groups[key].append(run_dir)
        
        for (prefix, norm_suffix), run_dirs in groups.items():
            if len(run_dirs) == 0:
                continue
            
            # If there's only one run and it's already normalized, we can skip if we want,
            # but we still might need to rename it if it's not normalized.
            
            # Find the one with best AUC
            best_run = None
            best_auc = -1.0
            
            run_info = [] # List of (path, auc)
            
            for run_path in run_dirs:
                score_file = run_path / "score.json"
                auc = -1.0
                if score_file.exists():
                    try:
                        with open(score_file, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                auc = data.get("auc", -1.0)
                    except Exception as e:
                        print(f"  Error reading {score_file}: {e}")
                
                run_info.append((run_path, auc))
                
                if auc > best_auc:
                    best_auc = auc
                    best_run = run_path
                elif auc == best_auc and best_run is not None:
                    # Preference: 
                    # 1. Already has the target name
                    # 2. Shorter name (usually the normalized one)
                    target_name = prefix + norm_suffix
                    if run_path.name == target_name:
                        best_run = run_path
                    elif best_run.name != target_name and len(run_path.name) < len(best_run.name):
                        best_run = run_path
            
            if best_run is None:
                best_run = run_dirs[0]

            # Delete others
            for run_path, auc in run_info:
                if run_path != best_run:
                    print(f"  Removing duplicate: {run_path.name} (AUC: {auc:.4f})")
                    # try:
                        # shutil.rmtree(run_path)
                    # except Exception as e:
                        # print(f"  Error removing {run_path}: {e}")
            
            # # Normalize the winner's name
            # target_name = prefix + norm_suffix
            # if best_run.name != target_name:
            #     target_path = best_run.parent / target_name
            #     if target_path.exists():
            #         print(f"  Warning: Target name {target_name} already exists (might be a different run). Skipping rename.")
            #     else:
            #         print(f"  Renaming {best_run.name} -> {target_name} (AUC: {best_auc:.4f})")
            #         try:
            #             best_run.rename(target_path)
            #         except Exception as e:
            #             print(f"  Error renaming {best_run.name} to {target_name}: {e}")

if __name__ == "__main__":
    base_dir = "/home/mpham/workspace/source/ehrfm/ehrpfn-inference/results/benchmark-reproduce/1603/openml_ehr/"
    clean_duplicates(base_dir)
