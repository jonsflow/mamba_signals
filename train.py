#!/usr/bin/env python3
"""Main training script for Mamba signals model."""
import sys
import logging
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config, default_config
from data_loader import OHLCDataLoader
from windowing import SequenceWindower, LabelGenerator, split_dataset
from training import Trainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    config = default_config
    db_path = Path(__file__).parent / "ohlc_data.db"

    logger.info("=" * 80)
    logger.info("MAMBA SIGNALS - TRAINING PIPELINE")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("\n1. Loading OHLC data...")
    with OHLCDataLoader(db_path, config.data) as loader:
        data_dict = loader.get_all_symbols_data()
        summary = loader.get_data_summary()

    logger.info(f"Loaded data for {len(data_dict)} symbols")

    # 2. Combine all symbols and normalize
    logger.info("\n2. Normalizing data...")
    all_windows = []
    all_labels = []

    for symbol, df in data_dict.items():
        logger.info(f"Processing {symbol}...")

        # Normalize
        with OHLCDataLoader(db_path, config.data) as loader:
            df_norm, scaling_params = loader.normalize_ohlcv(df)

        # Window
        windower = SequenceWindower(
            config.data.sequence_length,
            config.labeling.future_bars
        )
        windows, indices = windower.create_windows(df_norm)

        # Label
        label_gen = LabelGenerator(config.labeling, config.data.sequence_length)
        labels, valid_mask = label_gen.generate_labels(df, windows, indices)

        all_windows.append(windows)
        all_labels.append(labels)

        logger.info(f"  Windows: {len(windows)}, Valid: {valid_mask.sum()}")

    # Combine all symbols
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(f"\nCombined dataset:")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Labels distribution: {np.bincount(y)}")

    # 3. Split dataset
    logger.info("\n3. Splitting dataset...")
    mask = np.ones(len(X), dtype=bool)  # All valid in this case
    splits = split_dataset(
        X, y, mask,
        test_split=config.data.test_split,
        val_split=config.data.val_split,
        random_seed=config.training.random_seed
    )

    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    logger.info(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    logger.info(f"  Train labels: {np.bincount(y_train)}")
    logger.info(f"  Val labels: {np.bincount(y_val)}")
    logger.info(f"  Test labels: {np.bincount(y_test)}")

    # 4. Train model
    logger.info("\n4. Training model...")
    # M1 Mac support: use MPS if available, otherwise CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"  Device: Metal Performance Shaders (M1/M2/M3)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"  Device: CUDA GPU")
    else:
        device = torch.device('cpu')
        logger.info(f"  Device: CPU")
    logger.info(f"  Model: {config.model}")

    trainer = Trainer(config, model_dir=Path(__file__).parent / "models")
    trainer.fit((X_train, y_train), (X_val, y_val))

    # 5. Save results
    logger.info("\n5. Saving results...")
    trainer.save_history()

    # 6. Evaluate on test set
    logger.info("\n6. Evaluating on test set...")
    with torch.no_grad():
        trainer.model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        logits = trainer.model(X_test_tensor.to(trainer.device))
        preds = logits.argmax(dim=1)
        test_acc = (preds.cpu() == y_test_tensor).float().mean().item()

        logger.info(f"  Test Accuracy: {test_acc:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
