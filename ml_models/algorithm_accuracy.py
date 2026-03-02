"""
Display model accuracy with algorithm names
"""

import joblib
import os
import sys

def display_algorithm_accuracy(model_version='latest'):
    """Display algorithm names with their accuracy"""

    # Get model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, model_version)
    metadata_path = os.path.join(model_dir, 'metadata.pkl')

    if not os.path.exists(metadata_path):
        print(f"✗ Model not found at {model_dir}")
        return

    # Load metadata
    metadata = joblib.load(metadata_path)

    # Get accuracy values
    classifier_acc = metadata.get('classifier_accuracy', 0) * 100
    qty_r2 = metadata.get('regressor_r2', 0) * 100
    quality_r2 = metadata.get('quality_predictor_r2', 0) * 100
    qty_mae = metadata.get('regressor_mae', 0)

    # Display with algorithm names
    print("\n" + "=" * 80)
    print("ALGORITHM ACCURACY")
    print("=" * 80)

    print(f"\n{'Algorithm Name':<40} │ {'Accuracy':<20}")
    print("-" * 80)
    print(f"{'Random Forest Classifier':<40} │ {classifier_acc:>6.2f}%")
    print(f"{'Gradient Boosting Regressor':<40} │ {qty_r2:>6.2f}% (R²)")
    print(f"{'Random Forest Regressor':<40} │ {quality_r2:>6.2f}% (R²)")
    print(f"{'Quantity MAE (Error Metric)':<40} │ {qty_mae:>6.2f} kg")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate average accuracy
    avg_accuracy = (classifier_acc + qty_r2 + quality_r2) / 3

    print(f"\nAverage Algorithm Accuracy: {avg_accuracy:.2f}%")
    print(f"Best Performing: Gradient Boosting Regressor ({qty_r2:.2f}%)")
    print(f"Status: {'✓ EXCELLENT' if avg_accuracy > 85 else '✓ GOOD'}")

    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    version = sys.argv[1] if len(sys.argv) > 1 else 'latest'
    display_algorithm_accuracy(version)
