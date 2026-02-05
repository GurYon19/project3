"""
Restore dataset from backup.
"""
import shutil
from pathlib import Path

def restore_from_backup(data_dir: str):
    """Restore dataset from backup."""
    data_path = Path(data_dir)
    backup_path = data_path.parent / f"{data_path.name}_backup"
    
    if not backup_path.exists():
        print(f"❌ No backup found at: {backup_path}")
        return
    
    print(f"Restoring from: {backup_path}")
    print(f"To: {data_path}")
    
    # Remove current
    if data_path.exists():
        shutil.rmtree(data_path)
    
    # Restore
    shutil.copytree(backup_path, data_path)
    
    print("✅ Dataset restored!")

if __name__ == "__main__":
    restore_from_backup("datasets/part2")
