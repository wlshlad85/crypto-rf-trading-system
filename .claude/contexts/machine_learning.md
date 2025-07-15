# Machine Learning Context

## Overview
Advanced ensemble meta-learning system implementing 4-model Random Forest architecture with regime-aware optimization. Achieves 52% prediction accuracy through sophisticated feature engineering, cross-validation, and Hidden Markov Model integration for market regime detection.

## Critical Files
- `phase2b/ensemble_meta_learning.py` - Primary ML orchestration with 4-model ensemble
- `phase2b/hmm_regime_detection.py` - Market regime classification (97.9% Bull persistence)
- `models/enhanced_rf_models.pkl` - Production-trained ensemble models
- `models/random_forest_model.py` - Core Random Forest implementation
- `features/ultra_feature_engineering.py` - 78+ engineered features
- `phase1/cpcv_framework.py` - Combinatorial Purged Cross-Validation

## Key Classes and Functions

### EnhancedRFEnsemble
- **Purpose**: Meta-learning orchestration with 4 specialized Random Forest models
- **Key Methods**:
  - `train_ensemble_models()` - Train entry/position/exit/profit models independently
  - `predict_with_ensemble()` - Aggregate predictions with confidence scoring
  - `validate_cross_temporal()` - 51-window walk-forward validation
  - `optimize_hyperparameters()` - Hyperband-based parameter optimization
- **Architecture**: 4 models × 100 trees × depth 10 = 400 decision trees total

### HMMRegimeDetector
- **Purpose**: Market regime classification for strategy adaptation
- **Key Methods**:
  - `fit_regime_model()` - Train 3-state HMM (Bull/Bear/Sideways)
  - `predict_current_regime()` - Real-time regime classification
  - `get_regime_probabilities()` - Probabilistic state assessment
  - `analyze_regime_persistence()` - Regime stability analysis
- **Performance**: 97.9% Bull regime persistence, 94.2% Bear regime accuracy

### UltraFeatureEngineer
- **Purpose**: Advanced feature engineering with 78+ technical indicators
- **Key Methods**:
  - `engineer_all_features()` - Generate complete feature set
  - `select_optimal_features()` - Feature importance-based selection
  - `cross_timeframe_fusion()` - Multi-timeframe feature integration
  - `create_regime_features()` - Regime-specific feature engineering
- **Features**: Technical (42), Statistical (18), Regime (8), On-chain proxies (10)

## Model Architecture

### Ensemble Structure
```python
ensemble_models = {
    'entry_model': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
    ),
    'position_model': RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt'
    ),
    'exit_model': RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=25,
        min_samples_leaf=12,
        max_features='log2'
    ),
    'profit_model': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt'
    )
}
```

### Feature Engineering Pipeline
1. **Raw Data Processing**: OHLCV + Volume normalization
2. **Technical Indicators**: RSI, MACD, Bollinger Bands, Ichimoku Cloud
3. **Statistical Features**: Rolling statistics, volatility measures
4. **Cross-Timeframe**: 1m, 5m, 15m, 1h feature aggregation
5. **Regime Features**: HMM state probabilities and transitions
6. **On-chain Proxies**: MVRV, NVT, Exchange flow simulations

## Performance Metrics

### Current Model Performance
- **Ensemble Accuracy**: 52.1% (target: 55%+)
- **Entry Model**: 54.3% precision, 48.7% recall
- **Position Model**: R² = 0.34, RMSE = 0.12
- **Exit Model**: 51.8% precision, 56.2% recall
- **Profit Model**: R² = 0.28, RMSE = 0.15

### Cross-Validation Results
- **Walk-Forward Windows**: 51 validation periods
- **PBO Risk**: 22.3% (Probability of Backtest Overfitting)
- **Sharpe Consistency**: 78% of windows > 1.0 Sharpe
- **Drawdown Control**: 89% of windows < 15% max drawdown

### Regime Detection Performance
- **Bull Regime**: 97.9% persistence, 94.1% accuracy
- **Bear Regime**: 89.3% persistence, 91.7% accuracy  
- **Sideways Regime**: 72.4% persistence, 86.5% accuracy
- **Regime Transition Accuracy**: 84.2% next-state prediction

## Feature Importance Analysis

### Top 15 Features (Current Ensemble)
1. **rsi_14** (0.089) - 14-period Relative Strength Index
2. **regime_prob_bull** (0.082) - HMM Bull regime probability
3. **volume_sma_ratio** (0.076) - Volume to SMA ratio
4. **macd_signal_cross** (0.071) - MACD signal line crossover
5. **volatility_regime** (0.069) - Volatility clustering regime
6. **ichimoku_cloud_position** (0.065) - Price position relative to cloud
7. **momentum_1h** (0.063) - 1-hour momentum indicator
8. **bb_squeeze** (0.061) - Bollinger Band squeeze detection
9. **volume_profile_poc** (0.058) - Point of Control from volume profile
10. **fractional_diff_price** (0.056) - Fractionally differentiated price
11. **regime_transition_prob** (0.054) - Regime change probability
12. **multi_tf_consensus** (0.052) - Cross-timeframe signal consensus
13. **kelly_optimal_size** (0.051) - Kelly criterion position size
14. **cvar_risk_score** (0.049) - CVaR-based risk assessment
15. **onchain_mvrv_proxy** (0.047) - Simulated MVRV ratio

## Training Pipeline

### Data Preparation
1. **Quality Validation**: 99.5% data quality threshold
2. **Feature Engineering**: 78+ features across multiple timeframes
3. **Label Creation**: Triple barrier method with genetic optimization
4. **Train/Validation Split**: 80/20 with purged cross-validation
5. **Regime Stratification**: Ensure regime representation in splits

### Model Training Process
```python
def train_ensemble_pipeline():
    # 1. Feature engineering and selection
    features = engineer_features(raw_data)
    selected_features = select_features(features, importance_threshold=0.01)
    
    # 2. Regime-aware data splitting
    train_idx, val_idx = regime_stratified_split(features, test_size=0.2)
    
    # 3. Train individual models
    for model_name, model in ensemble_models.items():
        model.fit(X_train[selected_features], y_train[model_name])
        
    # 4. Meta-learning optimization
    optimize_ensemble_weights(models, validation_data)
    
    # 5. Final validation
    validate_ensemble_performance(models, holdout_data)
```

### Hyperparameter Optimization
- **Method**: Hyperband algorithm for efficient search
- **Search Space**: 15 hyperparameters per model
- **Budget**: 500 configurations × 4 models = 2000 total trials
- **Validation**: 5-fold cross-validation with temporal splits
- **Objective**: Multi-objective (accuracy + Sharpe + drawdown)

## Real-time Integration

### Live Prediction Pipeline
1. **Data Ingestion**: Real-time market data from enhanced_paper_trader_24h.py
2. **Feature Computation**: On-the-fly feature engineering (< 50ms)
3. **Regime Assessment**: Current HMM state probability
4. **Ensemble Prediction**: 4-model voting with confidence scoring
5. **Signal Generation**: Convert predictions to trading signals

### Performance Requirements
- **Prediction Latency**: < 100ms for ensemble prediction
- **Feature Engineering**: < 50ms for real-time features
- **Model Loading**: < 500ms for ensemble initialization
- **Memory Usage**: < 2GB for all models and features

### Current Live Performance
- **Signal Confidence**: 0.98-0.99 (SELL signals with high confidence)
- **Regime State**: Bull market (97.9% persistence)
- **Feature Quality**: All 78 features computing successfully
- **Prediction Latency**: 45ms average (well within 100ms target)

## Model Validation & Testing

### Cross-Validation Framework
- **CPCV Implementation**: Combinatorial Purged Cross-Validation
- **Purge Period**: 24 hours to prevent data leakage
- **Embargo Period**: 12 hours after purge
- **Number of Splits**: 51 walk-forward windows
- **Validation Metric**: Risk-adjusted returns (Sharpe ratio)

### Overfitting Prevention
- **Early Stopping**: Monitor validation loss during training
- **Regularization**: L2 penalty in meta-learning layer
- **Feature Selection**: Remove correlated features (r > 0.9)
- **Ensemble Diversity**: Different hyperparameters per model
- **Out-of-Sample Testing**: Final holdout set (6 months)

### Model Monitoring
- **Prediction Drift**: Track prediction distribution changes
- **Feature Importance Stability**: Monitor feature ranking changes
- **Performance Degradation**: Alert on accuracy drops > 5%
- **Regime Detection Accuracy**: Validate HMM predictions daily

## Dependencies & Requirements

### Core ML Libraries
- **scikit-learn**: Random Forest implementation and metrics
- **pandas/numpy**: Data manipulation and numerical computation
- **scipy**: Statistical functions and optimization
- **hmmlearn**: Hidden Markov Model implementation
- **joblib**: Model serialization and parallel processing

### Custom Modules
- **features/**: Feature engineering and selection
- **phase1/**: Cross-validation and data validation
- **utils/**: Configuration management and utilities
- **data/**: Market data feeding and preprocessing

## Configuration & Tuning

### Model Configuration Files
- `models/feature_config.json` - Feature engineering parameters
- `models/ensemble_config.json` - Model hyperparameters
- `models/regime_config.json` - HMM configuration
- `models/validation_config.json` - Cross-validation settings

### Key Parameters
```python
ML_CONFIG = {
    'ensemble': {
        'n_models': 4,
        'voting_method': 'weighted',
        'confidence_threshold': 0.6,
        'retraining_frequency': 'weekly'
    },
    'features': {
        'n_features': 78,
        'selection_method': 'importance',
        'correlation_threshold': 0.9,
        'regime_features': True
    },
    'validation': {
        'n_splits': 51,
        'purge_hours': 24,
        'embargo_hours': 12,
        'min_samples_per_split': 1000
    }
}
```

## Known Issues & Solutions
- **Feature Correlation**: High correlation between some technical indicators
  - Solution: Correlation-based feature selection (r < 0.9)
- **Regime Transition Lag**: HMM slow to detect regime changes
  - Solution: Ensemble with faster momentum indicators
- **Sample Imbalance**: Uneven distribution of Bull/Bear/Sideways periods
  - Solution: Stratified sampling and SMOTE for minority classes
- **Prediction Stability**: High variance in ensemble predictions
  - Solution: Increase ensemble size and implement prediction smoothing

## Future ML Enhancements
- [ ] LSTM neural networks for sequence modeling
- [ ] Transformer models for attention-based predictions
- [ ] Reinforcement learning for action optimization
- [ ] AutoML pipeline for automated feature engineering
- [ ] Multi-asset correlation modeling
- [ ] Alternative data integration (sentiment, news)

## Model Lifecycle Management
- **Training Schedule**: Weekly retraining with new data
- **Model Versioning**: Git-based model version control
- **A/B Testing**: Gradual rollout of new model versions
- **Performance Monitoring**: Real-time model performance dashboards
- **Rollback Procedures**: Automatic rollback on performance degradation

## Integration with Trading System
- **Signal Generation**: ML predictions → trading signals
- **Risk Integration**: Model confidence → position sizing
- **Performance Feedback**: Trading results → model retraining
- **Regime Adaptation**: HMM states → strategy selection
- **Feature Updates**: Market data → real-time feature engineering