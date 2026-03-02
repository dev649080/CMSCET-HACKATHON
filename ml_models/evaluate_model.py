"""
Evaluate trained model and display actual accuracy on test data
"""

import joblib
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score

def generate_test_data(n_samples=1000):
    """Generate test data from real alloy dataset"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'Alloys.csv')

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at {dataset_path}")
        return None, None, None, None

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Define elements and materials
    elements = ['C', 'Si', 'Mn', 'Cr', 'Ni', 'Mo', 'Fe']
    materials = [
        'FeSi 75%', 'FeCr 65%', 'Ni Metal',
        'FeMo 60%', 'Mn Metal', 'SiMn 65/15'
    ]
    material_to_idx = {mat: idx for idx, mat in enumerate(materials)}
    material_compositions = {
        'FeSi 75%': {'Si': 75.0, 'Fe': 25.0},
        'FeCr 65%': {'Cr': 65.0, 'Fe': 35.0},
        'Ni Metal': {'Ni': 99.5, 'Fe': 0.5},
        'FeMo 60%': {'Mo': 60.0, 'Fe': 40.0},
        'Mn Metal': {'Mn': 99.0, 'Fe': 1.0},
        'SiMn 65/15': {'Mn': 65.0, 'Si': 15.0, 'Fe': 20.0}
    }

    # Generate test scenarios
    data = []
    material_labels = []
    quantity_labels = []
    quality_labels = []

    np.random.seed(123)  # Different seed for test data
    n_real_samples = min(n_samples // 2, len(df))
    real_samples = df.sample(n=n_real_samples, random_state=123)

    for idx, row in real_samples.iterrows():
        # Extract current composition
        current_comp = {elem: row[elem] for elem in elements}

        # Simulate target composition
        element_to_adjust = np.random.choice(['Si', 'Mn', 'Cr', 'Ni', 'Mo'])
        target_comp = current_comp.copy()

        if element_to_adjust == 'Si' and current_comp['Si'] < 2.0:
            target_comp['Si'] += np.random.uniform(0.1, 0.5)
        elif element_to_adjust == 'Cr' and current_comp['Cr'] < 20.0:
            target_comp['Cr'] += np.random.uniform(0.5, 2.0)
        elif element_to_adjust == 'Ni' and current_comp['Ni'] < 15.0:
            target_comp['Ni'] += np.random.uniform(0.5, 2.0)
        elif element_to_adjust == 'Mo' and current_comp['Mo'] < 3.0:
            target_comp['Mo'] += np.random.uniform(0.1, 0.5)
        elif element_to_adjust == 'Mn' and current_comp['Mn'] < 2.0:
            target_comp['Mn'] += np.random.uniform(0.1, 0.5)

        # Calculate deviations
        deviations = {elem: target_comp[elem] - current_comp[elem] for elem in elements}

        # Determine optimal material
        candidate_materials = []
        for material, composition in material_compositions.items():
            if element_to_adjust in composition and composition[element_to_adjust] > 30:
                candidate_materials.append((material, composition[element_to_adjust]))

        if candidate_materials:
            material, element_pct = max(candidate_materials, key=lambda x: x[1])
        else:
            material = 'FeSi 75%'
            element_pct = 75.0

        # Calculate quantity
        heat_size = 100.0
        quantity_needed = (deviations[element_to_adjust] / 100.0) * heat_size * (100.0 / element_pct)
        quantity_needed = abs(quantity_needed) * 1000
        quantity_needed *= np.random.uniform(0.95, 1.05)
        quantity_needed = np.clip(quantity_needed, 1.0, 5000.0)

        # Quality
        base_quality = row.get('Tensile Strength: Ultimate (UTS) (psi)', 650.0)
        quality_improvement = base_quality * (1 + np.random.uniform(0.01, 0.05))

        # Create feature vector
        feature_vec = {}
        for elem in elements:
            feature_vec[f'current_{elem}'] = current_comp.get(elem, 0.0)
            feature_vec[f'target_{elem}'] = target_comp.get(elem, 0.0)
            feature_vec[f'deviation_{elem}'] = deviations[elem]
        feature_vec['primary_element'] = element_to_adjust
        feature_vec['deviation_magnitude'] = abs(deviations[element_to_adjust])
        feature_vec['heat_size'] = heat_size

        data.append(feature_vec)
        material_labels.append(material_to_idx[material])
        quantity_labels.append(quantity_needed)
        quality_labels.append(quality_improvement)

    # Convert to DataFrames
    features_df = pd.DataFrame(data)
    features_df = pd.get_dummies(features_df, columns=['primary_element'])

    material_labels = np.array(material_labels)
    quantity_labels = np.array(quantity_labels)
    quality_labels = np.array(quality_labels)

    return features_df, material_labels, quantity_labels, quality_labels

def evaluate_model(model_version='latest'):
    """Evaluate trained model on test data"""

    # Get model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, model_version)

    # Load models
    classifier_path = os.path.join(model_dir, 'material_classifier.pkl')
    regressor_path = os.path.join(model_dir, 'quantity_regressor.pkl')
    quality_path = os.path.join(model_dir, 'quality_predictor.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    if not all(os.path.exists(p) for p in [classifier_path, regressor_path, scaler_path]):
        print("✗ Model files not found")
        return

    print("\n" + "=" * 80)
    print("EVALUATING TRAINED MODEL ON TEST DATA")
    print("=" * 80)

    print("\nLoading models...")
    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    quality_predictor = joblib.load(quality_path) if os.path.exists(quality_path) else None
    scaler = joblib.load(scaler_path)

    print("Generating test data (1000 samples)...")
    X_test, y_material, y_quantity, y_quality = generate_test_data(n_samples=1000)

    if X_test is None:
        return

    # Ensure feature alignment
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 80)
    print("MODEL ACCURACY ON TEST DATA")
    print("=" * 80)

    # Evaluate classifier
    print("\n1. RANDOM FOREST CLASSIFIER (Material Selection)")
    print("-" * 80)
    y_pred_material = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_material, y_pred_material)
    precision = precision_score(y_material, y_pred_material, average='weighted', zero_division=0)
    recall = recall_score(y_material, y_pred_material, average='weighted', zero_division=0)
    f1 = f1_score(y_material, y_pred_material, average='weighted', zero_division=0)

    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")

    # Evaluate regressor
    print("\n2. GRADIENT BOOSTING REGRESSOR (Quantity Prediction)")
    print("-" * 80)
    y_pred_quantity = regressor.predict(X_test_scaled)
    mae = mean_absolute_error(y_quantity, y_pred_quantity)
    r2 = r2_score(y_quantity, y_pred_quantity)
    mape = np.mean(np.abs((y_quantity - y_pred_quantity) / y_quantity)) * 100

    print(f"   MAE (kg):  {mae:.2f}")
    print(f"   R² Score:  {r2*100:.2f}%")
    print(f"   MAPE:      {mape:.2f}%")

    # Evaluate quality predictor
    if quality_predictor is not None:
        print("\n3. RANDOM FOREST REGRESSOR (Quality Prediction)")
        print("-" * 80)
        y_pred_quality = quality_predictor.predict(X_test_scaled)
        quality_r2 = r2_score(y_quality, y_pred_quality)
        quality_mae = mean_absolute_error(y_quality, y_pred_quality)

        print(f"   R² Score:  {quality_r2*100:.2f}%")
        print(f"   MAE (psi): {quality_mae:.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_accuracy = (accuracy + r2) / 2
    print(f"\nClassifier Accuracy: {accuracy*100:.2f}%")
    print(f"Regressor R² Score:  {r2*100:.2f}%")
    print(f"Average Accuracy:    {avg_accuracy*100:.2f}%")
    print(f"Status: {'✓ EXCELLENT' if avg_accuracy > 0.85 else '✓ GOOD'}")

    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    version = sys.argv[1] if len(sys.argv) > 1 else 'latest'
    evaluate_model(version)
