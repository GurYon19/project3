import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Part 2: Single Object Detection Video Router")
    parser.add_argument("--video", "-v", type=str, help="Path to input video for tracking (e.g., 'part2_video/Big Cats.mp4')")
    parser.add_argument("--image", "-i", type=str, help="Path to static image (optional)")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/part2/best_model.pth", help="Model checkpoint path")
    args = parser.parse_args()

    print("==================================================")
    print("Running Part 2: Detection and EMA Video Tracking")
    print(f"Checkpoint: {args.checkpoint}")
    print("==================================================")
    
    if not os.path.exists(args.checkpoint):
        print(f"\n[!] Error: Checkpoint not found at '{args.checkpoint}'")
        print("Please ensure the checkpoint file exists in the submission zip.")
        return

    cmd = [sys.executable, "part2/inference.py", "--checkpoint", args.checkpoint]
    
    if args.video:
        cmd.extend(["--video", args.video])
        print(f"Tracking Video: {args.video}")
    elif args.image:
        cmd.extend(["--image", args.image])
        print(f"Tracking Image: {args.image}")
    else:
        print("\n[!] Please provide a target video or image.")
        print("Example: python part2_runner.py --video \"part2_video/your_video.mp4\"")
        return

    # Pass control to the core Part 2 inference script
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n[!] Execution failed or was interrupted.")
    else:
        print("\n[+] Part 2 Video Tracking Completed Successfully.")

if __name__ == "__main__":
    main()
