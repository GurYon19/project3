import subprocess
import sys

def main():
    print("==================================================")
    print("Running Part 1: Classification Inference Demo")
    print("==================================================")
    
    # Calls the actual implementation built for Part 1
    result = subprocess.run([sys.executable, "part1/train.py"])
    
    if result.returncode != 0:
        print("\n[!] Execution failed or was interrupted.")
    else:
        print("\n[+] Part 1 Execution Completed Successfully.")

if __name__ == "__main__":
    main()
