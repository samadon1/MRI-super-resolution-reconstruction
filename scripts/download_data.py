from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import DigitDataset

def main():
    """Download and process the digit dataset"""
    dataset = DigitDataset()
    dataset.process_and_save()
    processed_files = dataset.get_processed_files()
    print(f"\nTotal processed files: {len(processed_files)}")
    if processed_files:
        print(f"Files saved in: {dataset.processed_dir}")

if __name__ == "__main__":
    main()
