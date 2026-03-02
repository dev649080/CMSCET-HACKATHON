"""
Train AI-Driven Alloy Addition System Models
Trains 3 ML models for material classification, quantity prediction, and quality prediction
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def load_and_prepare_data():
    """Load alloys dataset and prepare features"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'Alloys.csv')

    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at {dataset_path}")
        return None, None, None, None

    print("📊 Loading Alloys dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   ✓ Loaded {len(df)} alloy records with {len(df.columns)} features")

    # Define working elements
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

    # Generate training scenarios from real alloys
    data = []
    material_labels = []
    quantity_labels = []
    quality_labels = []

    np.random.seed(42)  # Consistent seed for reproducibility

    print("\n🔄 Generating training scenarios from {} alloys...".format(len(df)))

    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"   Processing: {idx}/{len(df)}")

        # Extract current composition
        current_comp = {elem: row[elem] for elem in elements}

        # Randomly select element to adjust
        element_to_adjust = np.random.choice(['Si', 'Mn', 'Cr', 'Ni', 'Mo'])
        target_comp = current_comp.copy()

        # Simulate target composition variations
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

        # Calculate quantity needed
        heat_size = 100.0
        quantity_needed = (deviations[element_to_adjust] / 100.0) * heat_size * (100.0 / element_pct)
        quantity_needed = abs(quantity_needed) * 1000
        quantity_needed *= np.random.uniform(0.95, 1.05)
        quantity_needed = np.clip(quantity_needed, 1.0, 5000.0)

        # Quality metric
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

    print(f"\n   ✓ Generated {len(features_df)} training samples")
    print(f"   ✓ Feature matrix shape: {features_df.shape}")

    return features_df, material_labels, quantity_labels, quality_labels

def train_models():
    """Train all three ML models"""

    print("\n" + "="*80)
    print("AI-DRIVEN ALLOY ADDITION SYSTEM - MODEL TRAINING")
    print("="*80)

    # Load and prepare data
    X, y_material, y_quantity, y_quality = load_and_prepare_data()

    if X is None:
        print("✗ Failed to load data")
        return False

    # Split data
    print("\n📂 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_mat_train, y_mat_test, y_qty_train, y_qty_test, y_qual_train, y_qual_test = train_test_split(
        X, y_material, y_quantity, y_quality, test_size=0.2, random_state=42
    )
    print(f"   ✓ Train samples: {len(X_train)}")
    print(f"   ✓ Test samples: {len(X_test)}")

    # Scale features
    print("\n⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✓ StandardScaler fitted and applied")

    # ============ MODEL 1: MATERIAL CLASSIFIER ============
    print("\n" + "-"*80)
    print("MODEL 1: RANDOM FOREST CLASSIFIER (Material Selection)")
    print("-"*80)

    print("🤖 Training Random Forest Classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    classifier.fit(X_train_scaled, y_mat_train)

    # Predictions and metrics
    y_pred_material = classifier.predict(X_test_scaled)
    classifier_accuracy = accuracy_score(y_mat_test, y_pred_material)
    classifier_precision = precision_score(y_mat_test, y_pred_material, average='weighted', zero_division=0)
    classifier_recall = recall_score(y_mat_test, y_pred_material, average='weighted', zero_division=0)
    classifier_f1 = f1_score(y_mat_test, y_pred_material, average='weighted', zero_division=0)

    print(f"\n✓ Model trained successfully")
    print(f"  • Accuracy:  {classifier_accuracy*100:.2f}%")
    print(f"  • Precision: {classifier_precision*100:.2f}%")
    print(f"  • Recall:    {classifier_recall*100:.2f}%")
    print(f"  • F1-Score:  {classifier_f1*100:.2f}%")

    # ============ MODEL 2: QUANTITY REGRESSOR ============
    print("\n" + "-"*80)
    print("MODEL 2: GRADIENT BOOSTING REGRESSOR (Quantity Prediction)")
    print("-"*80)

    print("🤖 Training Gradient Boosting Regressor...")
    regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=0
    )
    regressor.fit(X_train_scaled, y_qty_train)

    # Predictions and metrics
    y_pred_quantity = regressor.predict(X_test_scaled)
    qty_mae = mean_absolute_error(y_qty_test, y_pred_quantity)
    qty_r2 = r2_score(y_qty_test, y_pred_quantity)
    qty_rmse = np.sqrt(mean_squared_error(y_qty_test, y_pred_quantity))
    qty_mape = np.mean(np.abs((y_qty_test - y_pred_quantity) / (y_qty_test + 1))) * 100

    print(f"\n✓ Model trained successfully")
    print(f"  • R² Score: {qty_r2*100:.2f}%")
    print(f"  • MAE:      {qty_mae:.2f} kg")
    print(f"  • RMSE:     {qty_rmse:.2f} kg")
    print(f"  • MAPE:     {qty_mape:.2f}%")

    # ============ MODEL 3: QUALITY PREDICTOR ============
    print("\n" + "-"*80)
    print("MODEL 3: RANDOM FOREST REGRESSOR (Quality Prediction)")
    print("-"*80)

    print("🤖 Training Random Forest Regressor...")
    quality_predictor = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    quality_predictor.fit(X_train_scaled, y_qual_train)

    # Predictions and metrics
    y_pred_quality = quality_predictor.predict(X_test_scaled)
    quality_r2 = r2_score(y_qual_test, y_pred_quality)
    quality_mae = mean_absolute_error(y_qual_test, y_pred_quality)
    quality_rmse = np.sqrt(mean_squared_error(y_qual_test, y_pred_quality))

    print(f"\n✓ Model trained successfully")
    print(f"  • R² Score: {quality_r2*100:.2f}%")
    print(f"  • MAE:      {quality_mae:.2f} psi")
    print(f"  • RMSE:     {quality_rmse:.2f} psi")

    # ============ SAVE MODELS ============
    print("\n" + "-"*80)
    print("SAVING MODELS")
    print("-"*80)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'latest')
    os.makedirs(model_dir, exist_ok=True)

    print(f"📁 Saving to: {model_dir}")

    # Save models
    joblib.dump(classifier, os.path.join(model_dir, 'material_classifier.pkl'))
    joblib.dump(regressor, os.path.join(model_dir, 'quantity_regressor.pkl'))
    joblib.dump(quality_predictor, os.path.join(model_dir, 'quality_predictor.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    print("   ✓ material_classifier.pkl")
    print("   ✓ quantity_regressor.pkl")
    print("   ✓ quality_predictor.pkl")
    print("   ✓ scaler.pkl")

    # Save metadata
    metadata = {
        'classifier_accuracy': classifier_accuracy,
        'classifier_precision': classifier_precision,
        'classifier_recall': classifier_recall,
        'classifier_f1': classifier_f1,
        'regressor_r2': qty_r2,
        'regressor_mae': qty_mae,
        'regressor_rmse': qty_rmse,
        'regressor_mape': qty_mape,
        'quality_predictor_r2': quality_r2,
        'quality_predictor_mae': quality_mae,
        'quality_predictor_rmse': quality_rmse,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X.shape[1],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    joblib.dump(metadata, os.path.join(model_dir, 'metadata.pkl'))
    print("   ✓ metadata.pkl")

    # ============ SUMMARY ============
    print("\n" + "="*80)
    print("TRAINING SUMMARY & ACCURACY REPORT")
    print("="*80)

    print("\n📊 OVERALL PERFORMANCE METRICS")
    print("-"*80)

    print("\n1️⃣  MATERIAL CLASSIFIER (Random Forest)")
    print(f"    Accuracy:  {classifier_accuracy*100:6.2f}%")
    print(f"    Precision: {classifier_precision*100:6.2f}%")
    print(f"    Recall:    {classifier_recall*100:6.2f}%")
    print(f"    F1-Score:  {classifier_f1*100:6.2f}%")

    print("\n2️⃣  QUANTITY REGRESSOR (Gradient Boosting)")
    print(f"    R² Score:  {qty_r2*100:6.2f}%")
    print(f"    MAE:       {qty_mae:6.2f} kg")
    print(f"    RMSE:      {qty_rmse:6.2f} kg")
    print(f"    MAPE:      {qty_mape:6.2f}%")

    print("\n3️⃣  QUALITY PREDICTOR (Random Forest)")
    print(f"    R² Score:  {quality_r2*100:6.2f}%")
    print(f"    MAE:       {quality_mae:6.2f} psi")
    print(f"    RMSE:      {quality_rmse:6.2f} psi")

    print("\n" + "-"*80)
    print("DATASET STATISTICS")
    print("-"*80)
    print(f"Training Samples:  {len(X_train):,}")
    print(f"Test Samples:      {len(X_test):,}")
    print(f"Total Features:    {X.shape[1]}")
    print(f"Material Classes:  6")
    print(f"Train-Test Split:  80-20")

    # Calculate composite accuracy
    composite_accuracy = (classifier_accuracy + qty_r2 + quality_r2) / 3

    print("\n" + "="*80)
    print(f"🎯 COMPOSITE ACCURACY: {composite_accuracy*100:.2f}%")
    print("="*80)

    if composite_accuracy > 0.90:
        print("✅ STATUS: EXCELLENT - Model ready for production deployment")
    elif composite_accuracy > 0.85:
        print("✅ STATUS: GOOD - Model suitable for production with monitoring")
    elif composite_accuracy > 0.80:
        print("⚠️  STATUS: ACCEPTABLE - Model needs tuning before production")
    else:
        print("❌ STATUS: NEEDS IMPROVEMENT - Recommend model optimization")

    print("\n" + "="*80 + "\n")

    return True

if __name__ == '__main__':
    success = train_models()
    sys.exit(0 if success else 1)
