import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageClassification, ViTImageProcessor
from datetime import datetime

# =========================
# CONFIG
# =========================
MODEL_NAME = "Falconsai/brand_identification"
VALID_BRANDS = {"adidas", "nike", "puma"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CONFIDENCE_THRESHOLD = 0  # 80% minimum certainty

# =========================
# PATHS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FAKE_DIR = PROJECT_ROOT / "Experiment Results" / "Fake Photos"
REAL_DIR = PROJECT_ROOT / "Experiment Results" / "Real Photos"

# =========================
# INITIALIZATION
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

print("Loading ViT Model...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()


# =========================
# HELPERS
# =========================
def normalize(name: str) -> str:
    if not name:
        return "unknown"
    return name.lower().strip().replace("addidas", "adidas")


def predict_vit_with_confidence(image_path: Path):
    """
    Predicts brand and returns (label, confidence_score).
    Returns (None, 0) if file is corrupt.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)

            # Apply Softmax to get probabilities
            probs = F.softmax(outputs.logits, dim=-1)
            conf, pred_id = torch.max(probs, dim=-1)

            confidence_score = conf.item()
            label = normalize(model.config.id2label[pred_id.item()])

            # Apply threshold logic
            if confidence_score < CONFIDENCE_THRESHOLD:
                return "unknown", confidence_score

            return label, confidence_score

    except (UnidentifiedImageError, OSError, Exception):
        return None, 0


# =========================
# CORE EXPERIMENT LOGIC
# =========================
def run_benchmark():
    results = {"total": 0, "correct": 0, "failed": [], "corrupt": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PROJECT_ROOT / f"model_AI_results_{timestamp}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"DNN BRAND DETECTION (Threshold: {CONFIDENCE_THRESHOLD})\n")
        f.write(f"Date: {timestamp} | Device: {device}\n")
        f.write("=" * 80 + "\n\n")

        # Function to process directories
        def process_dir(directory, expected_is_fake=False, folder_label=None):
            if not directory.exists():
                return

            images = [
                i for i in directory.glob("*") if i.suffix.lower() in IMAGE_EXTENSIONS
            ]
            for img_path in images:
                results["total"] += 1
                pred_label, confidence = predict_vit_with_confidence(img_path)

                if pred_label is None:
                    results["corrupt"].append(img_path.name)
                    f.write(f"{img_path.name:<35} -> [CORRUPT]\n")
                    continue

                # Determine correctness
                if expected_is_fake:
                    # For fake photos, correct if it returns 'unknown' or a non-brand
                    is_correct = (
                        pred_label == "unknown" or pred_label not in VALID_BRANDS
                    )
                    target = "NOT_BRAND"
                else:
                    is_correct = pred_label == folder_label
                    target = folder_label

                status = "CORRECT" if is_correct else "FAILED"
                if is_correct:
                    results["correct"] += 1
                else:
                    results["failed"].append((img_path.name, target, pred_label))

                f.write(
                    f"{img_path.name:<35} -> Pred: {pred_label:<12} ({confidence:.2%}) | {status}\n"
                )

        # Run Fake Tests
        f.write("SECTION: FAKE PHOTOS\n" + "-" * 40 + "\n")
        process_dir(FAKE_DIR, expected_is_fake=True)

        # Run Real Tests
        f.write("\nSECTION: REAL PHOTOS\n" + "-" * 40 + "\n")
        for brand_folder in REAL_DIR.iterdir():
            if brand_folder.is_dir():
                f.write(f"\n[{brand_folder.name.upper()}]\n")
                process_dir(
                    brand_folder,
                    expected_is_fake=False,
                    folder_label=normalize(brand_folder.name),
                )

        # Final Summary
        valid_count = results["total"] - len(results["corrupt"])
        acc = (results["correct"] / valid_count * 100) if valid_count > 0 else 0

        summary = (
            f"SUMMARY (Threshold: {CONFIDENCE_THRESHOLD})\n"
            f"Total Files:      {results['total']}\n"
            f"Corrupt Files:    {len(results['corrupt'])}\n"
            f"Valid Processed:  {valid_count}\n"
            f"Correct:          {results['correct']}\n"
            f"Accuracy:         {acc:.2f}%\n"
        )
        f.write(summary)
        print(summary)


if __name__ == "__main__":
    run_benchmark()
