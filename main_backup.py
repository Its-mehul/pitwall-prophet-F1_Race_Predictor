import random
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')


class PositionPredictionNetwork(nn.Module):
    """Neural network for predicting race finishing positions"""
    
    def __init__(self, input_dim, hidden_layers=(128, 64), dropout=0.2, activation='relu'):
        super().__init__()
        
        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        act_fn = activation_map.get(activation, nn.ReLU())
        
        # Per-driver feature encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                type(act_fn)(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.position_head = nn.Linear(prev_dim, 20)
        
    def forward(self, x):
        """x: [batch_size, num_drivers (20), num_features]"""
        batch_size, num_drivers, num_features = x.shape
        x_flat = x.view(-1, num_features)
        encoded = self.encoder(x_flat)
        logits = self.position_head(encoded)
        return logits.view(batch_size, num_drivers, 20)


class F1PositionPredictionPipeline:
    """End-to-end pipeline for F1 race position prediction"""
    
    def __init__(self, data_file, device='cpu', seed=339):
        self.data_file = data_file
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def load_data(self):
        """Load preprocessed race data"""
        print("="*80)
        print("Loading Preprocessed Data")
        print("="*80)
        
        data = np.load(self.data_file, allow_pickle=True)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.feature_names = data['feature_names']
        self.race_metadata = data['race_metadata']
        
        print(f"Training set: {self.X_train.shape[0]} races")
        print(f"Test set: {self.X_test.shape[0]} races")
        print(f"Features per driver: {self.X_train.shape[2]}\n")
    
    def create_mask(self, y):
        """Create mask for valid drivers (position <= 20)"""
        return (y <= 20).astype(np.float32)
    
    def hungarian_matching(self, pred_probs, mask=None):
        """Use Hungarian algorithm to assign unique positions to drivers"""
        batch_size, num_drivers, num_positions = pred_probs.shape
        assignments = np.zeros((batch_size, num_drivers), dtype=np.int32)
        
        for b in range(batch_size):
            cost_matrix = -pred_probs[b].cpu().detach().numpy()
            
            if mask is not None:
                valid_drivers = mask[b].cpu().numpy().astype(bool)
                cost_matrix[~valid_drivers] = 1e9
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments[b, row_ind] = col_ind + 1
        
        return assignments
    
    def compute_position_accuracy(self, predictions, targets, mask=None, tolerance=0):
        """Compute accuracy of position predictions with tolerance"""
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        valid_mask = target_valid <= 20
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        within_tolerance = np.abs(pred_valid - target_valid) <= tolerance
        correct = within_tolerance.sum()
        total = len(target_valid)
        
        return correct / total if total > 0 else 0.0
    
    def compute_bin_accuracy(self, predictions, targets, mask=None):
        """Compute accuracy using position bins (Podium/Points/Midfield/Back)"""
        def position_to_bin(pos):
            if pos <= 3:
                return 0
            elif pos <= 10:
                return 1
            elif pos <= 15:
                return 2
            else:
                return 3
        
        if mask is not None:
            valid = mask.astype(bool)
            pred_valid = predictions[valid]
            target_valid = targets[valid]
        else:
            pred_valid = predictions.flatten()
            target_valid = targets.flatten()
        
        valid_mask = target_valid <= 20
        pred_valid = pred_valid[valid_mask]
        target_valid = target_valid[valid_mask]
        
        pred_bins = np.array([position_to_bin(p) for p in pred_valid])
        target_bins = np.array([position_to_bin(t) for t in target_valid])
        
        correct = (pred_bins == target_bins).sum()
        total = len(target_valid)
        
        return correct / total if total > 0 else 0.0
    
    def train_single_fold(self, X_train_fold, y_train_fold, X_val_fold, y_val_fold, params):
        """Train model on a single fold and return validation accuracy"""
        train_mask = self.create_mask(y_train_fold)
        val_mask = self.create_mask(y_val_fold)
        
        X_train_t = torch.from_numpy(X_train_fold).float()
        y_train_t = torch.from_numpy(y_train_fold).long()
        train_mask_t = torch.from_numpy(train_mask).float()
        
        X_val_t = torch.from_numpy(X_val_fold).float()
        y_val_t = torch.from_numpy(y_val_fold).long()
        val_mask_t = torch.from_numpy(val_mask).float()
        
        train_ds = TensorDataset(X_train_t, y_train_t, train_mask_t)
        val_ds = TensorDataset(X_val_t, y_val_t, val_mask_t)
        
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False)
        
        input_dim = X_train_fold.shape[2]
        model = PositionPredictionNetwork(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation=params['activation']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                momentum=0.9
            )
        
        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            for xb, yb, mask_b in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mask_b = mask_b.to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(xb)
                logits_flat = logits.view(-1, 20)
                targets_flat = (yb - 1).clamp(0, 19).view(-1).long()
                mask_flat = mask_b.view(-1)
                
                loss_per_sample = criterion(logits_flat, targets_flat)
                loss = (loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation fold
        model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for xb, yb, mask_b in val_loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=-1)
                
                assignments = self.hungarian_matching(probs, mask_b)
                
                all_preds.append(assignments)
                all_targets.append(yb.numpy())
                all_masks.append(mask_b.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        tol2_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=2)
        return tol2_acc
    
    def cross_validate(self, params, n_folds=5):
        """Perform k-fold cross-validation and return average accuracy"""
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        fold_accs = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]
            
            fold_acc = self.train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold, params
            )
            fold_accs.append(fold_acc)
        
        return np.mean(fold_accs)
    
    def grid_search(self, param_grid):
        """Perform grid search with 5-fold cross-validation"""
        print("="*80)
        print("Starting Grid Search with 5-Fold Cross-Validation")
        print("="*80)
        
        keys = param_grid.keys()
        values = param_grid.values()
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        total_combinations = len(param_combinations)
        print(f"\nTotal parameter combinations to test: {total_combinations}\n")
        
        best_score = -1
        best_params = None
        
        for i, params in enumerate(param_combinations, 1):
            print(f"[{i}/{total_combinations}] Testing: "
                  f"layers={params['hidden_layers']}, "
                  f"dropout={params['dropout']}, "
                  f"act={params['activation']}, "
                  f"lr={params['lr']}, "
                  f"opt={params['optimizer']}, "
                  f"bs={params['batch_size']}, "
                  f"wd={params['weight_decay']}, "
                  f"epochs={params['epochs']}")
            
            try:
                avg_tol2_acc = self.cross_validate(params)
                print(f"  → CV Accuracy (±2 positions): {avg_tol2_acc*100:.2f}%")
                
                if avg_tol2_acc > best_score:
                    best_score = avg_tol2_acc
                    best_params = params.copy()
                    print(f"  ✓ New best!")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print("\n" + "="*80)
        print("Grid Search Complete!")
        print("="*80)
        print(f"\nBest parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nBest CV Accuracy (±2 positions): {best_score*100:.2f}%")
        print("="*80 + "\n")
        
        self.best_params = best_params
        self.best_cv_score = best_score
        
        return best_params, best_score
    
    def train_final_model(self, params):
        """Train final model on all training data"""
        print("\nTraining final model on all training data...")
        print(f"  Hidden layers: {params['hidden_layers']}")
        print(f"  Dropout: {params['dropout']}")
        print(f"  Activation: {params['activation']}")
        print(f"  Learning rate: {params['lr']}")
        print(f"  Optimizer: {params['optimizer']}")
        print(f"  Batch size: {params['batch_size']}")
        print(f"  Weight decay: {params['weight_decay']}")
        print(f"  Epochs: {params['epochs']}")
        print(f"  Device: {self.device}\n")
        
        train_mask = self.create_mask(self.y_train)
        
        X_train_t = torch.from_numpy(self.X_train).float()
        y_train_t = torch.from_numpy(self.y_train).long()
        train_mask_t = torch.from_numpy(train_mask).float()
        
        train_ds = TensorDataset(X_train_t, y_train_t, train_mask_t)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        
        input_dim = self.X_train.shape[2]
        self.model = PositionPredictionNetwork(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation=params['activation']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                momentum=0.9
            )
        
        # Training loop
        for epoch in range(1, params['epochs'] + 1):
            self.model.train()
            train_losses = []
            
            for xb, yb, mask_b in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mask_b = mask_b.to(self.device)
                
                optimizer.zero_grad()
                
                logits = self.model(xb)
                logits_flat = logits.view(-1, 20)
                targets_flat = (yb - 1).clamp(0, 19).view(-1).long()
                mask_flat = mask_b.view(-1)
                
                loss_per_sample = criterion(logits_flat, targets_flat)
                loss = (loss_per_sample * mask_flat).sum() / (mask_flat.sum() + 1e-8)
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            
            if epoch == 1 or epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f}")
        
        return self.model
    
    def test_model(self, show_predictions=True):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print("Final Test Performance")
        print("="*80)
        
        raw_df = pd.read_csv('f1_raw_2021_2025.csv')
        
        test_mask = self.create_mask(self.y_test)
        
        X_test_t = torch.from_numpy(self.X_test).float()
        y_test_t = torch.from_numpy(self.y_test).long()
        test_mask_t = torch.from_numpy(test_mask).float()
        
        test_ds = TensorDataset(X_test_t, y_test_t, test_mask_t)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for xb, yb, mask_b in test_loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=-1)
                
                assignments = self.hungarian_matching(probs, mask_b)
                
                all_preds.append(assignments)
                all_targets.append(yb.numpy())
                all_masks.append(mask_b.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        # Print detailed predictions for each race
        if show_predictions:
            total_races = len(self.race_metadata)
            test_start_idx = total_races - len(all_preds)
            
            for i in range(len(all_preds)):
                race_idx = test_start_idx + i
                race_meta = self.race_metadata[race_idx]
                
                print(f"\n{'='*80}")
                print(f"Race {i+1}: {race_meta['year']} Round {race_meta['round']}")
                print(f"{race_meta['race_name'][:70]}")
                print(f"{'='*80}")
                
                race_df = raw_df[
                    (raw_df['year'] == int(race_meta['year'])) & 
                    (raw_df['round'] == int(race_meta['round'])) &
                    (raw_df['final_position'].notna())
                ].sort_values('final_position').reset_index(drop=True)
                
                race_preds = all_preds[i]
                race_targets = all_targets[i]
                race_mask = all_masks[i]
                
                # Collect predictions
                predictions = []
                for pos in range(1, 21):
                    idx = np.where(race_targets == pos)[0]
                    if len(idx) > 0 and race_mask[idx[0]] == 1:
                        idx = idx[0]
                        driver_name = race_df.iloc[idx]['driver_name'] if idx < len(race_df) else 'Unknown'
                        team_name = race_df.iloc[idx]['team_name'] if idx < len(race_df) else 'Unknown'
                        pred_pos = int(race_preds[idx])
                        diff = abs(pred_pos - pos)
                        
                        predictions.append({
                            'pos': pos,
                            'driver': driver_name,
                            'team': team_name,
                            'predicted': pred_pos,
                            'diff': diff
                        })
                    else:
                        break
                
                # Print grouped by accuracy
                sections = [
                    ('✓ EXACT PREDICTIONS', lambda p: p['diff'] == 0),
                    ('~ OFF BY 1 POSITION', lambda p: p['diff'] == 1),
                    ('~ OFF BY 2 POSITIONS', lambda p: p['diff'] == 2),
                    ('~ OFF BY 3 POSITIONS', lambda p: p['diff'] == 3),
                    ('✗ OFF BY 4+ POSITIONS', lambda p: p['diff'] >= 4)
                ]
                
                for section_title, filter_func in sections:
                    section_preds = [p for p in predictions if filter_func(p)]
                    if section_preds:
                        print(f"\n{section_title}:")
                        print(f"{'Pos':<5} {'Driver':<25} {'Team':<30} {'Predicted':<12}")
                        print("-"*80)
                        for p in section_preds:
                            print(f"P{p['pos']:<3} {p['driver'][:24]:<25} {p['team'][:29]:<30} "
                                  f"P{p['predicted']:<10}")
                
                # Race metrics
                exact = self.compute_position_accuracy(race_preds.reshape(1, -1), race_targets.reshape(1, -1), 
                                                       race_mask.reshape(1, -1), tolerance=0)
                tol2 = self.compute_position_accuracy(race_preds.reshape(1, -1), race_targets.reshape(1, -1), 
                                                      race_mask.reshape(1, -1), tolerance=2)
                valid = (race_targets <= 20) & (race_mask == 1)
                race_mae = mean_absolute_error(race_targets[valid], race_preds[valid]) if valid.sum() > 0 else 0.0
                
                print("-"*80)
                print(f"Race Accuracy: Exact={exact*100:.1f}%, ±2 positions={tol2*100:.1f}%, MAE={race_mae:.2f}")
        
        # Overall metrics
        exact_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=0)
        tol1_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=1)
        tol2_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=2)
        tol3_acc = self.compute_position_accuracy(all_preds, all_targets, all_masks, tolerance=3)
        bin_acc = self.compute_bin_accuracy(all_preds, all_targets, all_masks)
        
        valid_mask = (all_targets <= 20) & (all_masks == 1)
        mae = mean_absolute_error(
            all_targets[valid_mask].flatten(),
            all_preds[valid_mask].flatten()
        )
        
        print(f"\n{'='*80}")
        print(f"Overall Test Performance ({len(all_preds)} races):")
        print(f"{'='*80}")
        print(f"Test Exact Position Accuracy: {exact_acc * 100:.2f}%")
        print(f"Test Accuracy (±1 position): {tol1_acc * 100:.2f}%")
        print(f"Test Accuracy (±2 positions): {tol2_acc * 100:.2f}%")
        print(f"Test Accuracy (±3 positions): {tol3_acc * 100:.2f}%")
        print(f"Test Bin Accuracy (Podium/Points/Mid/Back): {bin_acc * 100:.2f}%")
        print(f"Test Mean Absolute Error: {mae:.2f} positions")
        print("="*80 + "\n")
        
        self.test_tol2_acc = tol2_acc
        self.test_mae = mae
        
        return tol2_acc, mae


def main():
    """Main execution pipeline"""
    # Configuration
    DATA_FILE = "f1_processed_2021_2025.npz"
    RUN_GRID_SEARCH = False
    
    # Initialize pipeline
    pipeline = F1PositionPredictionPipeline(
        data_file=DATA_FILE,
        device='cuda',
        seed=339
    )
    
    # Load data
    pipeline.load_data()
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_layers': [(128, 64), (64, 32)],
        'dropout': [0.2],
        'activation': ['relu', 'leaky_relu'],
        'lr': [1e-3, 5e-4],
        'optimizer': ['adam'],
        'batch_size': [16, 32],
        'weight_decay': [1e-4],
        'epochs': [50]
    }
    
    if RUN_GRID_SEARCH:
        # Grid search with cross-validation
        best_params, best_cv_score = pipeline.grid_search(param_grid)
    else:
        # Use preset best parameters
        best_params = {
            'hidden_layers': (64, 32),
            'dropout': 0.2,
            'activation': 'relu',
            'lr': 1e-3,
            'optimizer': 'adam',
            'batch_size': 32,
            'weight_decay': 1e-4,
            'epochs': 50
        }
        print("\nUsing best hyperparameters from previous grid search")
    
    # Train final model
    pipeline.train_final_model(best_params)
    
    # Test on held-out test set
    pipeline.test_model(show_predictions=True)


if __name__ == "__main__":
    main()
