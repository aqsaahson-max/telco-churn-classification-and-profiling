"""Restore omitted data from the owner's original archive; never retrain or overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_project', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    split = json.loads((root/'data/splits/split_record.json').read_text())
    frozen = json.loads((root/'models/frozen_model_procedure.json').read_text())
    evaluation = json.loads((root/'outputs/tables/final_evaluation/final_evaluation_run_record.json').read_text())
    expected = {
        'data/raw/' + split['source_filename']: split['source_sha256'],
        'data/splits/development.csv': frozen['development_sha256'],
        'data/splits/test.csv': evaluation['test_file_sha256'],
    }
    # Validate every source/destination before copying anything.
    for rel, sha in expected.items():
        src, dst = args.original_project/rel, root/rel
        if not src.is_file() or digest(src) != sha:
            raise ValueError('Missing or mismatched original file: ' + rel)
        if dst.exists() and digest(dst) != sha:
            raise FileExistsError('Refusing to replace a different file: ' + rel)
    for rel, sha in expected.items():
        src, dst = args.original_project/rel, root/rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        if digest(dst) != sha:
            raise ValueError('Post-copy fingerprint mismatch: ' + rel)
        print('Verified:', rel)
    print('Original data restored. No model was loaded or fitted.')

if __name__ == '__main__':
    main()
