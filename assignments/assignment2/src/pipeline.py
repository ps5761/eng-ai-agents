
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_video import download_video
from extract_frames import extract_frames
from detect import load_model, detect_all_frames
from retrieve import run_retrieval_on_hf_dataset


STEPS = {
    "download": download_video,
    "extract": extract_frames,
    "detect": lambda: detect_all_frames(load_model()),
    "retrieve": run_retrieval_on_hf_dataset,
}


def run_all():
    print("=" * 60)
    print("  ASSIGNMENT 2: CAR PARTS RETRIEVAL PIPELINE")
    print("=" * 60)
    for name, fn in STEPS.items():
        print(f"\n{'=' * 60}")
        print(f"  STEP: {name}")
        print(f"{'=' * 60}")
        fn()
    print("\n[pipeline] All steps complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step in STEPS:
            STEPS[step]()
        else:
            print(f"Unknown step: {step}")
            print(f"Options: {', '.join(STEPS.keys())}")
            sys.exit(1)
    else:
        run_all()