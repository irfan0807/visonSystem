#!/usr/bin/env python3
"""
Audio Classification Model Training Script

Trains an SVM classifier for audio event detection.
Supports training on custom datasets or using sample data.

Usage:
    python audio_train.py --data-dir data/audio --output models/audio_classifier.pkl
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Audio classes for classification
CLASSES = [
    'normal',
    'speech', 
    'scream',
    'glass_break',
    'gunshot',
    'door_slam',
    'dog_bark',
    'car_horn',
    'siren'
]


def check_dependencies():
    """Check required dependencies."""
    missing = []
    if not HAS_LIBROSA:
        missing.append('librosa')
    if not HAS_SKLEARN:
        missing.append('scikit-learn')
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


def extract_features(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract audio features from a sample.
    
    Features:
    - MFCC (13 coefficients + delta)
    - Spectral Centroid
    - Spectral Rolloff
    - Spectral Bandwidth
    - Zero Crossing Rate
    - RMS Energy
    
    Args:
        audio: Audio samples
        sr: Sample rate
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i, mfcc_mean in enumerate(np.mean(mfccs, axis=1)):
        features[f'mfcc_{i}'] = float(mfcc_mean)
    for i, mfcc_std in enumerate(np.std(mfccs, axis=1)):
        features[f'mfcc_std_{i}'] = float(mfcc_std)
    
    # MFCC Delta
    mfcc_delta = librosa.feature.delta(mfccs)
    for i, delta_mean in enumerate(np.mean(mfcc_delta, axis=1)):
        features[f'mfcc_delta_{i}'] = float(delta_mean)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid'] = float(np.mean(spectral_centroid))
    features['spectral_centroid_std'] = float(np.std(spectral_centroid))
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
    features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
    features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)
    features['rms'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i, chroma_mean in enumerate(np.mean(chroma, axis=1)):
        features[f'chroma_{i}'] = float(chroma_mean)
    
    return features


def load_audio_file(
    file_path: Path, 
    sr: int = 16000, 
    duration: float = 2.0
) -> Optional[np.ndarray]:
    """
    Load an audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Duration to load (seconds)
    
    Returns:
        Audio samples or None if failed
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_dataset(
    data_dir: Path, 
    sr: int = 16000
) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Load dataset from directory structure.
    
    Expected structure:
    data_dir/
        class1/
            sample1.wav
            sample2.wav
        class2/
            sample1.wav
            ...
    
    Args:
        data_dir: Root data directory
        sr: Sample rate
    
    Returns:
        Tuple of (features_list, labels_list)
    """
    features_list = []
    labels_list = []
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in CLASSES:
            print(f"Skipping unknown class: {class_name}")
            continue
        
        print(f"Loading class: {class_name}")
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
        
        for audio_file in audio_files:
            audio = load_audio_file(audio_file, sr=sr)
            if audio is not None and len(audio) > sr // 2:  # Min 0.5s
                features = extract_features(audio, sr)
                features_list.append(features)
                labels_list.append(class_name)
        
        print(f"  Loaded {len([l for l in labels_list if l == class_name])} samples")
    
    return features_list, labels_list


def generate_synthetic_data(
    n_samples_per_class: int = 100,
    sr: int = 16000,
    duration: float = 2.0
) -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Generate synthetic training data for demonstration.
    
    Args:
        n_samples_per_class: Samples per class
        sr: Sample rate
        duration: Duration of each sample
    
    Returns:
        Tuple of (features_list, labels_list)
    """
    print("Generating synthetic training data...")
    
    features_list = []
    labels_list = []
    
    n_samples = int(sr * duration)
    
    # Generate samples for each class with different characteristics
    class_params = {
        'normal': {'freq_range': (200, 500), 'amplitude': 0.1, 'noise': 0.05},
        'speech': {'freq_range': (100, 400), 'amplitude': 0.3, 'noise': 0.1},
        'scream': {'freq_range': (800, 2000), 'amplitude': 0.9, 'noise': 0.2},
        'glass_break': {'freq_range': (2000, 8000), 'amplitude': 0.8, 'noise': 0.4},
        'gunshot': {'freq_range': (50, 200), 'amplitude': 1.0, 'noise': 0.3},
        'door_slam': {'freq_range': (100, 300), 'amplitude': 0.7, 'noise': 0.2},
        'dog_bark': {'freq_range': (400, 1200), 'amplitude': 0.6, 'noise': 0.15},
        'car_horn': {'freq_range': (300, 600), 'amplitude': 0.8, 'noise': 0.1},
        'siren': {'freq_range': (600, 1400), 'amplitude': 0.85, 'noise': 0.15},
    }
    
    for class_name, params in class_params.items():
        print(f"Generating {class_name} samples...")
        
        for i in range(n_samples_per_class):
            # Generate base signal
            t = np.linspace(0, duration, n_samples)
            freq = np.random.uniform(*params['freq_range'])
            amplitude = params['amplitude'] * np.random.uniform(0.8, 1.2)
            
            # Create signal with harmonics
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            signal += 0.3 * amplitude * np.sin(2 * np.pi * freq * 2 * t)  # Harmonic
            signal += 0.1 * amplitude * np.sin(2 * np.pi * freq * 3 * t)  # Harmonic
            
            # Add class-specific characteristics
            if class_name == 'scream':
                # Add frequency modulation
                mod_freq = np.random.uniform(2, 5)
                signal *= (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))
            
            elif class_name == 'glass_break':
                # Add attack transient
                envelope = np.exp(-3 * t)
                signal *= envelope
            
            elif class_name == 'gunshot':
                # Short impulse
                envelope = np.exp(-20 * t)
                signal *= envelope
            
            elif class_name == 'siren':
                # Oscillating frequency
                mod_freq = np.random.uniform(0.5, 2)
                freq_mod = 1 + 0.3 * np.sin(2 * np.pi * mod_freq * t)
                signal = amplitude * np.sin(2 * np.pi * freq * freq_mod * t)
            
            # Add noise
            noise = params['noise'] * np.random.randn(n_samples)
            signal += noise
            
            # Normalize
            signal = signal / (np.max(np.abs(signal)) + 1e-6)
            
            # Add some random variation
            signal *= np.random.uniform(0.5, 1.0)
            
            # Extract features
            features = extract_features(signal.astype(np.float32), sr)
            features_list.append(features)
            labels_list.append(class_name)
    
    return features_list, labels_list


def prepare_data(
    features_list: List[Dict[str, float]], 
    labels_list: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for training.
    
    Args:
        features_list: List of feature dictionaries
        labels_list: List of labels
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Get sorted feature names
    feature_names = sorted(features_list[0].keys())
    
    # Build feature matrix
    X = np.array([
        [features[f] for f in feature_names]
        for features in features_list
    ])
    
    y = np.array(labels_list)
    
    return X, y, feature_names


def train_model(
    X: np.ndarray, 
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[SVC, StandardScaler, Dict]:
    """
    Train the SVM classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        Tuple of (model, scaler, metrics)
    """
    print(f"\nTraining on {len(y)} samples...")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=random_state
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Compute metrics
    metrics = {
        'accuracy': float(model.score(X_test_scaled, y_test)),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'classes': list(model.classes_)
    }
    
    return model, scaler, metrics


def save_model(
    model: SVC, 
    scaler: StandardScaler, 
    feature_names: List[str],
    metrics: Dict,
    output_path: Path
):
    """
    Save trained model to file.
    
    Args:
        model: Trained classifier
        scaler: Feature scaler
        feature_names: List of feature names
        metrics: Training metrics
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'classifier': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'classes': CLASSES
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train audio classification model"
    )
    parser.add_argument(
        '--data-dir', type=Path, default=None,
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output', type=Path, 
        default=Path('models/audio_classifier.pkl'),
        help='Output model path'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Generate synthetic training data'
    )
    parser.add_argument(
        '--samples-per-class', type=int, default=100,
        help='Samples per class for synthetic data'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=16000,
        help='Audio sample rate'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Load or generate data
    if args.data_dir and args.data_dir.exists():
        features_list, labels_list = load_dataset(args.data_dir, args.sample_rate)
    else:
        if not args.synthetic:
            print("No data directory specified. Using synthetic data.")
            print("For custom training, use: --data-dir /path/to/audio/data")
        features_list, labels_list = generate_synthetic_data(
            n_samples_per_class=args.samples_per_class,
            sr=args.sample_rate
        )
    
    if len(features_list) == 0:
        print("No training data available!")
        sys.exit(1)
    
    # Prepare data
    X, y, feature_names = prepare_data(features_list, labels_list)
    
    # Train model
    model, scaler, metrics = train_model(X, y)
    
    # Save model
    save_model(model, scaler, feature_names, metrics, args.output)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
