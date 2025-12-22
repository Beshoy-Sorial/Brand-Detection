import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import logo_verifer
import sift


def load_logos(base_folder):
    """Load all logo images paths"""
    logos_dict = {}
    if isinstance(base_folder, str):
        base_path = Path(base_folder)
    else:
        base_path = base_folder

    if not base_path.exists():
        return {}

    for brand_folder in base_path.iterdir():
        if brand_folder.is_dir():
            brand_name = brand_folder.name
            logos = [
                str(file)
                for file in brand_folder.glob("*")
                if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
            logos_dict[brand_name] = logos

    return logos_dict


def normalize_brand_name(name):
    """Normalize brand names for comparison"""
    return name.lower().strip().replace("addidas", "adidas")


def test_dataset():
    """Test the brand detection on the experimental dataset"""

    # Initialize SIFT
    print("Initializing SIFT...")
    sift.init()

    # Determine base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent if script_dir.name == "src" else script_dir
    print(f"Base directory: {base_dir}")

    # Load logos
    logos_folder = base_dir / "Logos"
    if not logos_folder.exists():
        print(f"ERROR: Logos folder not found at {logos_folder}")
        return None, 0

    logos_paths = load_logos(logos_folder)
    descriptores = {}

    for brand, paths in logos_paths.items():
        descriptores[brand] = []
        print(f"Processing {len(paths)} logos for {brand}...")
        for path in paths:
            logo = cv2.imread(path)
            if logo is None:
                continue
            gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            kp_logo, des_logo = sift.calc_sift(gray_logo)
            descriptores[brand].append((kp_logo, des_logo))

    # Prepare test directories
    fake_folder = base_dir / "Experiment Results" / "Fake Photos"
    real_folder = base_dir / "Experiment Results" / "Real Photos"

    results = {"total": 0, "correct": 0, "failed": [], "succeeded": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_dir / f"Brand_Detection_results_{timestamp}.txt"

    # OPENING WITH buffering=1 forces line-by-line writing
    with open(log_file, "w", encoding="utf-8", buffering=1) as f:
        f.write("=" * 80 + "\n")
        f.write("BRAND DETECTION TEST RESULTS\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.flush()  # Force write header

        # --- Test Fake Photos ---
        if fake_folder.exists():
            print("\nTesting fake photos...")
            f.write("\nFAKE PHOTOS (Expected: UNKNOWN BRAND)\n" + "=" * 40 + "\n")

            fake_images = list(fake_folder.glob("*.png")) + list(
                fake_folder.glob("*.jpg")
            )
            for img_path in fake_images:
                results["total"] += 1
                image = cv2.imread(str(img_path))

                try:
                    result = logo_verifer.verify_logo(image, descriptores)
                    expected = "UNKNOWN BRAND"
                    is_correct = result == expected
                    status = "CORRECT" if is_correct else "FAILED"

                    log_entry = (
                        f"File: {img_path.name} | Got: {result} | Status: {status}\n"
                    )
                    f.write(log_entry)
                    f.flush()  # Ensure it saves immediately

                    if is_correct:
                        results["correct"] += 1
                        results["succeeded"].append(str(img_path))
                    else:
                        results["failed"].append(
                            {"file": str(img_path), "expected": expected, "got": result}
                        )
                except Exception as e:
                    f.write(f"ERROR processing {img_path.name}: {str(e)}\n")

        # --- Test Real Photos ---
        if real_folder.exists():
            print("\nTesting real photos...")
            for brand_folder in real_folder.iterdir():
                if not brand_folder.is_dir():
                    continue

                expected_brand = normalize_brand_name(brand_folder.name)
                f.write(
                    f"\nBRAND CATEGORY: {brand_folder.name.upper()}\n" + "-" * 40 + "\n"
                )

                brand_images = list(brand_folder.glob("*.png")) + list(
                    brand_folder.glob("*.jpg")
                )
                for img_path in brand_images:
                    results["total"] += 1
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    try:
                        result = logo_verifer.verify_logo(image, descriptores)
                        result_norm = normalize_brand_name(result)
                        is_correct = result_norm == expected_brand
                        status = "CORRECT" if is_correct else "FAILED"

                        log_entry = f"File: {img_path.name} | Got: {result} | Status: {status}\n"
                        f.write(log_entry)
                        f.flush()

                        if is_correct:
                            results["correct"] += 1
                        else:
                            results["failed"].append(
                                {
                                    "file": str(img_path),
                                    "expected": brand_folder.name,
                                    "got": result,
                                }
                            )
                    except Exception as e:
                        f.write(f"ERROR processing {img_path.name}: {str(e)}\n")

        # --- Final Summary ---
        accuracy = (
            (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0
        )
        summary = (
            f"\n{'='*80}\nSUMMARY\n{'='*80}\n"
            f"Total: {results['total']}\nCorrect: {results['correct']}\n"
            f"Accuracy: {accuracy:.2f}%\n"
        )
        f.write(summary)
        f.flush()
        print(summary)

    return results, accuracy


if __name__ == "__main__":
    test_dataset()
