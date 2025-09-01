
# IMPROVED TRAINING CONFIGURATION
class ImprovedConfig:
    # Hyperparameters (optimized)
    learning_rate = 0.0001
    batch_size = 16
    max_epochs = 100
    early_stopping_patience = 20
    
    # Model architecture
    cnn_channels = [64, 128, 256]  # Deeper
    gru_hidden = 256  # Larger
    gru_layers = 3  # Deeper
    dropout = 0.2
    use_attention = True
    
    # Training strategy
    optimizer = 'AdamW'
    weight_decay = 1e-5
    gradient_clip = 1.0
    scheduler = 'CosineAnnealingLR'
    warmup_epochs = 5
    label_smoothing = 0.1
    
    # Data augmentation
    augmentation_factor = 10
    augmentation_methods = [
        'gaussian_noise',
        'baseline_wander',
        'amplitude_scaling',
        'time_warping',
        'mixup'
    ]
    
    # Feature extraction
    use_morphological_features = True
    use_spectral_features = True
    use_hrv_features = True

# IMPROVED MODEL ARCHITECTURE
class ImprovedHybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Enhanced CNN branches
        self.cnn_branch1 = self._create_cnn_branch([3, 5, 7], config.cnn_channels)
        self.cnn_branch2 = self._create_cnn_branch([11, 15, 19], config.cnn_channels)
        
        # Enhanced GRU with attention
        self.gru = nn.GRU(
            input_size + len(config.cnn_channels) * 2,
            config.gru_hidden,
            config.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )
        
        # Self-attention mechanism
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                config.gru_hidden * 2,
                num_heads=8,
                dropout=config.dropout
            )
        
        # Enhanced fusion layers with residual connections
        self.fusion = nn.Sequential(
            nn.Linear(config.gru_hidden * 2 + len(config.cnn_channels) * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(128, 1)
        )
    
    def _create_cnn_branch(self, kernel_sizes, channels):
        layers = []
        in_channels = 1
        
        for i, (kernel_size, out_channels) in enumerate(zip(kernel_sizes, channels)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.LayerNorm(out_channels),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)

# IMPROVED TRAINING LOOP
def train_improved_model(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
        eta_min=1e-6
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config.warmup_epochs
    )
    
    criterion = nn.SmoothL1Loss()  # More robust than MSE
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Mixup augmentation
            if config.use_mixup and np.random.random() < 0.5:
                mixed_x, mixed_y = mixup(batch.x, batch.y, alpha=0.2)
                outputs = model(mixed_x)
                loss = criterion(outputs, mixed_y)
            else:
                outputs = model(batch.x)
                loss = criterion(outputs, batch.y)
            
            # Label smoothing
            if config.label_smoothing > 0:
                loss = (1 - config.label_smoothing) * loss +                        config.label_smoothing * loss.mean()
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_mae = evaluate_model(model, val_loader)
        
        # Learning rate scheduling
        if epoch < config.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break
    
    return model

# ENSEMBLE PREDICTION
def ensemble_predict(models, x):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
