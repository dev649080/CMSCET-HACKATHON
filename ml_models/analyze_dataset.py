"""
Analyze the Alloys.csv dataset
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('Alloys.csv')

print('='*60)
print('DATASET ANALYSIS')
print('='*60)
print(f'\nTotal samples: {len(df)}')
print(f'Total features: {len(df.columns)}')

print('\n' + '='*60)
print('MISSING VALUES')
print('='*60)
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print('No missing values!')

print('\n' + '='*60)
print('KEY ELEMENT COMPOSITION RANGES')
print('='*60)
elements = ['C', 'Si', 'Mn', 'Cr', 'Ni', 'Mo', 'Fe']
for elem in elements:
    print(f'{elem:3s}: min={df[elem].min():6.3f}%, max={df[elem].max():6.3f}%, avg={df[elem].mean():6.3f}%')

print('\n' + '='*60)
print('ALLOY TYPES DISTRIBUTION')
print('='*60)
print(df['Alloy'].value_counts().head(10))

print('\n' + '='*60)
print('MECHANICAL PROPERTIES')
print('='*60)
print(f"Tensile Strength (UTS):")
print(f"  Min: {df['Tensile Strength: Ultimate (UTS) (psi)'].min():.0f} psi")
print(f"  Max: {df['Tensile Strength: Ultimate (UTS) (psi)'].max():.0f} psi")
print(f"  Avg: {df['Tensile Strength: Ultimate (UTS) (psi)'].mean():.0f} psi")

print(f"\nMelting Point (Liquidus):")
print(f"  Min: {df['Melting Completion (Liquidus)'].min():.0f}°F")
print(f"  Max: {df['Melting Completion (Liquidus)'].max():.0f}°F")
print(f"  Avg: {df['Melting Completion (Liquidus)'].mean():.0f}°F")

print('\n' + '='*60)
print('HIGH-ALLOY CONTENT SAMPLES')
print('='*60)
high_cr = df[df['Cr'] > 15].shape[0]
high_ni = df[df['Ni'] > 8].shape[0]
high_mo = df[df['Mo'] > 2].shape[0]
print(f'High Chromium (>15%): {high_cr} samples ({high_cr/len(df)*100:.1f}%)')
print(f'High Nickel (>8%): {high_ni} samples ({high_ni/len(df)*100:.1f}%)')
print(f'High Molybdenum (>2%): {high_mo} samples ({high_mo/len(df)*100:.1f}%)')

print('\n' + '='*60)
print('STAINLESS STEEL DETECTION')
print('='*60)
stainless = df[(df['Cr'] > 10) & (df['Ni'] > 5)].shape[0]
print(f'Potential stainless steel alloys: {stainless} samples ({stainless/len(df)*100:.1f}%)')
