import os
import re
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

def find_log_files(root_dir: Path, note_substring: str) -> list[Path]:
    """
    Finds all 'train.log' files within subdirectories of root_dir that match the note.
    """
    log_files = []
    print(f"Searching for directories in '{root_dir}' containing note: '{note_substring}'")
    
    for subdir in root_dir.iterdir():
        if subdir.is_dir() and note_substring in subdir.name:
            log_file = subdir / 'train.log'
            if log_file.is_file():
                log_files.append(log_file)
                print(f"  [Found] {log_file}")
            else:
                print(f"  [Warning] Found matching directory, but no 'train.log': {subdir}")
                
    return log_files

def parse_log_files(log_paths: list[Path]) -> defaultdict[int, dict[str, float]]:
    """
    Parses a list of log files to extract test_domain accuracies for each domain and round.

    Args:
        log_paths: A list of paths to the train.log files.

    Returns:
        A dictionary mapping round number to another dictionary of {domain_name: accuracy}.
        Example: {0: {'p': 94.3, 'a': 83.8, ...}, 1: {...}}
    """
    ### MODIFIED ### - Regex now captures the domain name as group 2
    log_pattern = re.compile(r"test_domain Round:\s+(\d+).*Domain:\s+([pacrsvl]).*Acc:\s+([\d.]+)%")
    
    # Data structure now maps round -> domain -> accuracy
    round_to_domain_accuracies = defaultdict(dict)
    
    for log_path in log_paths:
        print(f"Processing log file: {log_path.parent.name}...")
        with open(log_path, 'r') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    round_num = int(match.group(1))
                    domain_name = str(match.group(2))
                    accuracy = float(match.group(3))
                    round_to_domain_accuracies[round_num][domain_name] = accuracy
                    
    return round_to_domain_accuracies

def main():
    """
    Main function to orchestrate the log analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze Fed-SC logs to find the best communication round.")
    parser.add_argument(
        "--output_folder", 
        type=str, 
        required=True, 
        help="The root output directory containing all experiment folders."
    )
    parser.add_argument(
        "--note", 
        type=str, 
        required=True, 
        help="The unique note string shared by the 4 experiment folders."
    )
    args = parser.parse_args()

    root_path = Path(args.output_folder)
    if not root_path.is_dir():
        print(f"Error: Provided output folder does not exist: {root_path}")
        return

    log_files = find_log_files(root_path, args.note)
    
    if len(log_files) != 4:
        print(f"\nError: Expected to find 4 log files for the note '{args.note}', but found {len(log_files)}.")
        print("Please ensure all 4 jobs have completed and folders are named correctly.")
        return
        
    all_accuracies = parse_log_files(log_files)
    
    if not all_accuracies:
        print("\nError: No 'test_domain' accuracy lines were found in the log files.")
        return

    average_accuracies = {}
    
    print("\n--- Average Accuracy per Communication Round ---")
    
    ### MODIFIED ### - Loop now prints per-domain results before the average
    for round_num in sorted(all_accuracies.keys()):
        domain_accs = all_accuracies[round_num]
        
        if len(domain_accs) < 4:
            print(f"  [Warning] Round {round_num}: Found only {len(domain_accs)}/4 domain results.")
        
        # Create a formatted string for the individual domain results
        # Sorting by domain name (a, c, p, r) for consistent output
        details = ", ".join([f"{domain.upper()}: {acc:.2f}%" for domain, acc in sorted(domain_accs.items())])
        
        acc_values = list(domain_accs.values())
        avg_acc = np.mean(acc_values)
        average_accuracies[round_num] = avg_acc
        
        # Print the detailed breakdown for the round
        print(f"Round {round_num:2d}: Avg Acc: {avg_acc:.2f}%  | Details: [{details}]")
        
    best_round = max(average_accuracies, key=average_accuracies.get)
    max_avg_acc = average_accuracies[best_round]
    
    # Get the detailed accuracies for the best round
    best_round_details_dict = all_accuracies[best_round]
    best_round_details_str = ", ".join([f"{domain.upper()}: {acc:.2f}%" for domain, acc in sorted(best_round_details_dict.items())])

    print("\n" + "="*65)
    print("                      ANALYSIS COMPLETE")
    print("="*65)
    print(f"PEAK AVERAGE ACCURACY: {max_avg_acc:.2f}%")
    print(f"BEST COMMUNICATION ROUND: {best_round}")
    print(f"Accuracies at Best Round: [{best_round_details_str}]")
    print("="*65)

if __name__ == "__main__":
    main()