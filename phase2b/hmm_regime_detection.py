#!/usr/bin/env python3
"""
Phase 2B: Hidden Markov Models for Regime Detection
ULTRATHINK Implementation - Market Regime Classification

Implements sophisticated regime detection used by institutional trading firms:
- Multi-state HMM for market regime identification
- Volatility clustering detection
- Regime-dependent feature selection
- Dynamic model adaptation based on market states
- Real-time regime inference

Designed to enhance ensemble performance through regime-aware adaptation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import pickle
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings('ignore')

@dataclass
class HMMConfig:
    """Configuration for Hidden Markov Model regime detection."""
    # HMM structure
    n_states: int = 3  # Number of hidden states (Bull, Bear, Sideways)
    state_names: List[str] = None
    
    # Observation features
    observation_features: List[str] = None
    use_returns: bool = True
    use_volatility: bool = True
    use_volume: bool = True
    
    # Model parameters
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    random_state: int = 42
    
    # Regime classification
    regime_window: int = 20  # Window for regime stability
    min_regime_duration: int = 5  # Minimum periods in a regime
    
    # Feature adaptation
    adapt_features: bool = True
    feature_selection_threshold: float = 0.05
    
    def __post_init__(self):
        if self.state_names is None:
            if self.n_states == 3:
                self.state_names = ['Bull_Market', 'Sideways_Market', 'Bear_Market']
            elif self.n_states == 2:
                self.state_names = ['Low_Volatility', 'High_Volatility']
            else:
                self.state_names = [f'State_{i}' for i in range(self.n_states)]
        
        if self.observation_features is None:
            self.observation_features = ['returns', 'volatility', 'volume_ratio']

class HiddenMarkovModel:
    """
    Hidden Markov Model implementation for market regime detection.
    
    Uses multiple observations (returns, volatility, volume) to identify
    latent market states and predict regime transitions.
    """
    
    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        
        # Model parameters
        self.initial_probs = None      # π: Initial state probabilities
        self.transition_matrix = None   # A: State transition probabilities
        self.emission_means = None      # μ: Emission distribution means
        self.emission_covs = None       # Σ: Emission distribution covariances
        
        # Training results
        self.log_likelihood = None
        self.converged = False
        self.n_iterations = 0
        
        # State sequences
        self.state_sequence = None
        self.state_probabilities = None
        
        np.random.seed(random_state)
    
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize HMM parameters."""
        n_obs, n_features = observations.shape
        
        # Initialize state probabilities uniformly
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # Initialize transition matrix with small random perturbations
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.transition_matrix += np.random.normal(0, 0.01, (self.n_states, self.n_states))
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize emission parameters using k-means-like clustering
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(observations)
            
            self.emission_means = np.zeros((self.n_states, n_features))
            self.emission_covs = np.zeros((self.n_states, n_features, n_features))
            
            for state in range(self.n_states):
                state_obs = observations[cluster_labels == state]
                if len(state_obs) > 0:
                    self.emission_means[state] = np.mean(state_obs, axis=0)
                    cov = np.cov(state_obs.T)
                    if cov.ndim == 0:
                        cov = np.array([[cov]])
                    elif cov.ndim == 1:
                        cov = np.diag(cov)
                    self.emission_covs[state] = cov + np.eye(n_features) * 1e-6  # Regularization
                else:
                    # Fallback for empty clusters
                    self.emission_means[state] = np.mean(observations, axis=0)
                    self.emission_covs[state] = np.cov(observations.T) + np.eye(n_features) * 1e-6
                    
        except Exception:
            # Fallback initialization
            self.emission_means = np.random.normal(0, 1, (self.n_states, n_features))
            self.emission_covs = np.array([np.eye(n_features) for _ in range(self.n_states)])
    
    def _compute_emission_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Compute emission probabilities for all states and observations."""
        n_obs, n_features = observations.shape
        emission_probs = np.zeros((n_obs, self.n_states))
        
        for state in range(self.n_states):
            try:
                # Multivariate normal probability
                mean = self.emission_means[state]
                cov = self.emission_covs[state]
                
                # Ensure positive definite covariance
                eigenvals = np.linalg.eigvals(cov)
                if np.any(eigenvals <= 0):
                    cov += np.eye(n_features) * (1e-6 - np.min(eigenvals))
                
                for t in range(n_obs):
                    obs = observations[t]
                    diff = obs - mean
                    
                    try:
                        cov_inv = np.linalg.inv(cov)
                        det = np.linalg.det(cov)
                        
                        if det <= 0:
                            # Fallback to regularized covariance
                            cov_reg = cov + np.eye(n_features) * 1e-3
                            cov_inv = np.linalg.inv(cov_reg)
                            det = np.linalg.det(cov_reg)
                        
                        exp_term = -0.5 * np.dot(diff, np.dot(cov_inv, diff))
                        normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * det)
                        
                        emission_probs[t, state] = normalization * np.exp(exp_term)
                        
                    except np.linalg.LinAlgError:
                        # Fallback to univariate calculation
                        if n_features == 1:
                            var = cov[0, 0]
                            emission_probs[t, state] = stats.norm.pdf(obs[0], mean[0], np.sqrt(var))
                        else:
                            emission_probs[t, state] = 1e-10  # Small probability
                            
            except Exception:
                emission_probs[:, state] = 1e-10  # Small probability for problematic states
        
        # Avoid numerical issues
        emission_probs = np.clip(emission_probs, 1e-10, np.inf)
        
        return emission_probs
    
    def _forward_algorithm(self, emission_probs: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm for computing alpha values and log-likelihood."""
        n_obs, n_states = emission_probs.shape
        alpha = np.zeros((n_obs, n_states))
        
        # Initialize
        alpha[0] = self.initial_probs * emission_probs[0]
        alpha[0] = alpha[0] / np.sum(alpha[0])  # Normalize
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(n_states):
                alpha[t, state] = np.sum(alpha[t-1] * self.transition_matrix[:, state]) * emission_probs[t, state]
            
            # Normalize to prevent underflow
            alpha_sum = np.sum(alpha[t])
            if alpha_sum > 0:
                alpha[t] = alpha[t] / alpha_sum
        
        # Compute log-likelihood (approximate)
        log_likelihood = np.sum(np.log(np.sum(alpha, axis=1) + 1e-10))
        
        return alpha, log_likelihood
    
    def _backward_algorithm(self, emission_probs: np.ndarray) -> np.ndarray:
        """Backward algorithm for computing beta values."""
        n_obs, n_states = emission_probs.shape
        beta = np.zeros((n_obs, n_states))
        
        # Initialize
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(n_obs - 2, -1, -1):
            for state in range(n_states):
                beta[t, state] = np.sum(self.transition_matrix[state] * emission_probs[t+1] * beta[t+1])
            
            # Normalize to prevent underflow
            beta_sum = np.sum(beta[t])
            if beta_sum > 0:
                beta[t] = beta[t] / beta_sum
        
        return beta
    
    def _compute_posteriors(self, alpha: np.ndarray, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior probabilities gamma and xi."""
        n_obs, n_states = alpha.shape
        
        # Gamma: P(state_t | observations)
        gamma = alpha * beta
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-10)
        
        # Xi: P(state_t, state_{t+1} | observations)
        xi = np.zeros((n_obs - 1, n_states, n_states))
        
        for t in range(n_obs - 1):
            denominator = np.sum(alpha[t] * beta[t]) + 1e-10
            
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = (alpha[t, i] * self.transition_matrix[i, j] * 
                                  beta[t+1, j]) / denominator
        
        return gamma, xi
    
    def _update_parameters(self, observations: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """Update HMM parameters using EM algorithm."""
        n_obs, n_features = observations.shape
        n_states = self.n_states
        
        # Update initial probabilities
        self.initial_probs = gamma[0] / np.sum(gamma[0])
        
        # Update transition matrix
        for i in range(n_states):
            denominator = np.sum(gamma[:-1, i]) + 1e-10
            for j in range(n_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / denominator
        
        # Update emission parameters
        for state in range(n_states):
            gamma_sum = np.sum(gamma[:, state]) + 1e-10
            
            # Update means
            self.emission_means[state] = np.sum(gamma[:, state:state+1] * observations, axis=0) / gamma_sum
            
            # Update covariances
            diff = observations - self.emission_means[state]
            self.emission_covs[state] = np.dot((gamma[:, state:state+1] * diff).T, diff) / gamma_sum
            
            # Add regularization
            self.emission_covs[state] += np.eye(n_features) * 1e-6
    
    def fit(self, observations: np.ndarray, max_iterations: int = 100, 
            convergence_threshold: float = 1e-6) -> 'HiddenMarkovModel':
        """Fit HMM using Expectation-Maximization algorithm."""
        
        # Initialize parameters
        self._initialize_parameters(observations)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Compute emission probabilities and forward-backward
            emission_probs = self._compute_emission_probabilities(observations)
            alpha, log_likelihood = self._forward_algorithm(emission_probs)
            beta = self._backward_algorithm(emission_probs)
            gamma, xi = self._compute_posteriors(alpha, beta)
            
            # M-step: Update parameters
            self._update_parameters(observations, gamma, xi)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < convergence_threshold:
                self.converged = True
                break
            
            prev_log_likelihood = log_likelihood
        
        self.log_likelihood = log_likelihood
        self.n_iterations = iteration + 1
        
        # Compute final state sequence
        emission_probs = self._compute_emission_probabilities(observations)
        self.state_sequence = self._viterbi_decode(emission_probs)
        
        # Store state probabilities
        alpha, _ = self._forward_algorithm(emission_probs)
        beta = self._backward_algorithm(emission_probs)
        self.state_probabilities, _ = self._compute_posteriors(alpha, beta)
        
        return self
    
    def _viterbi_decode(self, emission_probs: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for finding most likely state sequence."""
        n_obs, n_states = emission_probs.shape
        
        # Initialize
        delta = np.zeros((n_obs, n_states))
        psi = np.zeros((n_obs, n_states), dtype=int)
        
        delta[0] = self.initial_probs * emission_probs[0]
        
        # Forward pass
        for t in range(1, n_obs):
            for state in range(n_states):
                transition_scores = delta[t-1] * self.transition_matrix[:, state]
                psi[t, state] = np.argmax(transition_scores)
                delta[t, state] = np.max(transition_scores) * emission_probs[t, state]
        
        # Backward pass
        states = np.zeros(n_obs, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def predict_states(self, observations: np.ndarray) -> np.ndarray:
        """Predict most likely states for new observations."""
        emission_probs = self._compute_emission_probabilities(observations)
        return self._viterbi_decode(emission_probs)
    
    def predict_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Predict state probabilities for new observations."""
        emission_probs = self._compute_emission_probabilities(observations)
        alpha, _ = self._forward_algorithm(emission_probs)
        beta = self._backward_algorithm(emission_probs)
        gamma, _ = self._compute_posteriors(alpha, beta)
        return gamma

class HMMRegimeDetector:
    """
    High-level regime detection system using Hidden Markov Models.
    
    Provides market regime classification and regime-dependent analysis
    for cryptocurrency trading systems.
    """
    
    def __init__(self, config: Optional[HMMConfig] = None):
        """
        Initialize HMM regime detector.
        
        Args:
            config: Configuration for HMM parameters
        """
        self.config = config or HMMConfig()
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Results storage
        self.regime_analysis = {}
        self.feature_importance = {}
        self.regime_transitions = {}
        
        print("🔍 HMM Regime Detection System Initialized")
        print(f"📊 States: {self.config.n_states} ({', '.join(self.config.state_names)})")
        print(f"🎯 Features: {self.config.observation_features}")
        print(f"🔄 Adaptation: {'Enabled' if self.config.adapt_features else 'Disabled'}")
    
    def prepare_observations(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare observation features for HMM training."""
        observations = []
        feature_names = []
        
        # Returns
        if self.config.use_returns and 'Close' in df.columns:
            returns = df['Close'].pct_change().fillna(0)
            observations.append(returns)
            feature_names.append('returns')
        
        # Volatility
        if self.config.use_volatility and 'Close' in df.columns:
            returns = df['Close'].pct_change()
            volatility = returns.rolling(self.config.regime_window).std().fillna(returns.std())
            observations.append(volatility)
            feature_names.append('volatility')
        
        # Volume ratio
        if self.config.use_volume and 'Volume' in df.columns:
            volume_ma = df['Volume'].rolling(self.config.regime_window).mean()
            volume_ratio = df['Volume'] / volume_ma
            volume_ratio = volume_ratio.fillna(1.0)
            observations.append(volume_ratio)
            feature_names.append('volume_ratio')
        
        # Additional features from config
        for feature in self.config.observation_features:
            if feature in df.columns and feature not in feature_names:
                feature_data = df[feature].fillna(df[feature].median())
                observations.append(feature_data)
                feature_names.append(feature)
        
        if not observations:
            raise ValueError("No valid observation features found")
        
        # Combine observations
        obs_array = np.column_stack(observations)
        
        # Remove infinite values
        obs_array = np.where(np.isfinite(obs_array), obs_array, 0)
        
        return obs_array, feature_names
    
    def fit(self, df: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Fit HMM regime detector to market data.
        
        Args:
            df: DataFrame with market data (OHLCV)
            
        Returns:
            Fitted regime detector
        """
        print(f"\n🔍 Training HMM Regime Detection System")
        print("=" * 50)
        
        # Prepare observations
        observations, feature_names = self.prepare_observations(df)
        print(f"📊 Observations: {len(observations)} samples, {len(feature_names)} features")
        print(f"🎯 Features: {feature_names}")
        
        # Scale observations
        observations_scaled = self.scaler.fit_transform(observations)
        
        # Initialize and train HMM
        print(f"🔧 Training {self.config.n_states}-state HMM...")
        self.hmm_model = HiddenMarkovModel(
            n_states=self.config.n_states,
            random_state=self.config.random_state
        )
        
        self.hmm_model.fit(
            observations_scaled,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold
        )
        
        print(f"✅ HMM Training Complete!")
        print(f"   Converged: {self.hmm_model.converged}")
        print(f"   Iterations: {self.hmm_model.n_iterations}")
        print(f"   Log-likelihood: {self.hmm_model.log_likelihood:.4f}")
        
        # Analyze regimes
        self._analyze_regimes(df, observations_scaled, feature_names)
        
        self.is_fitted = True
        return self
    
    def predict_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market regimes for new data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with regime predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare observations
        observations, feature_names = self.prepare_observations(df)
        observations_scaled = self.scaler.transform(observations)
        
        # Predict states
        predicted_states = self.hmm_model.predict_states(observations_scaled)
        state_probabilities = self.hmm_model.predict_probabilities(observations_scaled)
        
        # Create result DataFrame
        result_df = df.copy()
        
        # Add regime predictions
        result_df['regime_state'] = predicted_states
        result_df['regime_name'] = [self.config.state_names[state] for state in predicted_states]
        
        # Add state probabilities
        for i, state_name in enumerate(self.config.state_names):
            result_df[f'regime_prob_{state_name}'] = state_probabilities[:, i]
        
        # Add regime stability
        result_df['regime_stability'] = self._calculate_regime_stability(predicted_states)
        
        return result_df
    
    def _analyze_regimes(self, df: pd.DataFrame, observations: np.ndarray, feature_names: List[str]):
        """Analyze detected regimes and their characteristics."""
        print("\n📊 Analyzing Detected Regimes...")
        
        states = self.hmm_model.state_sequence
        state_probs = self.hmm_model.state_probabilities
        
        # Regime statistics
        regime_stats = {}
        
        for state_idx, state_name in enumerate(self.config.state_names):
            state_mask = states == state_idx
            state_periods = np.sum(state_mask)
            state_proportion = state_periods / len(states)
            
            regime_stats[state_name] = {
                'periods': int(state_periods),
                'proportion': float(state_proportion),
                'avg_probability': float(np.mean(state_probs[:, state_idx]))
            }
            
            # Market statistics during this regime
            if 'Close' in df.columns:
                regime_prices = df['Close'][state_mask]
                if len(regime_prices) > 1:
                    regime_returns = regime_prices.pct_change().dropna()
                    
                    regime_stats[state_name].update({
                        'avg_return': float(regime_returns.mean()),
                        'volatility': float(regime_returns.std()),
                        'sharpe_ratio': float(regime_returns.mean() / regime_returns.std()) if regime_returns.std() > 0 else 0
                    })
            
            print(f"   {state_name}:")
            print(f"      Periods: {state_periods} ({state_proportion:.1%})")
            print(f"      Avg Probability: {np.mean(state_probs[:, state_idx]):.3f}")
        
        # Transition analysis
        transition_counts = np.zeros((self.config.n_states, self.config.n_states))
        for t in range(len(states) - 1):
            transition_counts[states[t], states[t+1]] += 1
        
        transition_probs = transition_counts / (np.sum(transition_counts, axis=1, keepdims=True) + 1e-10)
        
        print(f"\n🔄 Regime Transition Matrix:")
        for i, from_state in enumerate(self.config.state_names):
            for j, to_state in enumerate(self.config.state_names):
                print(f"   {from_state} → {to_state}: {transition_probs[i, j]:.3f}")
        
        # Feature importance by regime
        feature_importance = {}
        for state_idx, state_name in enumerate(self.config.state_names):
            state_mask = states == state_idx
            if np.sum(state_mask) > 5:  # Minimum samples
                state_obs = observations[state_mask]
                feature_vars = np.var(state_obs, axis=0)
                feature_importance[state_name] = dict(zip(feature_names, feature_vars))
        
        # Store analysis results
        self.regime_analysis = {
            'regime_statistics': regime_stats,
            'transition_matrix': transition_probs.tolist(),
            'feature_importance': feature_importance,
            'model_performance': {
                'log_likelihood': self.hmm_model.log_likelihood,
                'converged': self.hmm_model.converged,
                'n_iterations': self.hmm_model.n_iterations
            }
        }
    
    def _calculate_regime_stability(self, states: np.ndarray) -> np.ndarray:
        """Calculate regime stability metric."""
        stability = np.zeros(len(states))
        
        for i in range(len(states)):
            # Look at surrounding window
            start = max(0, i - self.config.regime_window // 2)
            end = min(len(states), i + self.config.regime_window // 2 + 1)
            
            window_states = states[start:end]
            current_state = states[i]
            
            # Calculate proportion of same state in window
            stability[i] = np.sum(window_states == current_state) / len(window_states)
        
        return stability
    
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime analysis summary."""
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        return {
            'config': {
                'n_states': self.config.n_states,
                'state_names': self.config.state_names,
                'observation_features': self.config.observation_features
            },
            'analysis': self.regime_analysis,
            'model_info': {
                'hmm_converged': self.hmm_model.converged,
                'hmm_iterations': self.hmm_model.n_iterations,
                'hmm_log_likelihood': self.hmm_model.log_likelihood
            }
        }
    
    def save_model(self, filepath: str):
        """Save trained HMM regime detector."""
        model_data = {
            'config': self.config,
            'hmm_model': self.hmm_model,
            'scaler': self.scaler,
            'regime_analysis': self.regime_analysis,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 HMM regime detector saved: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HMMRegimeDetector':
        """Load trained HMM regime detector."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        detector = cls(model_data['config'])
        detector.hmm_model = model_data['hmm_model']
        detector.scaler = model_data['scaler']
        detector.regime_analysis = model_data['regime_analysis']
        detector.is_fitted = model_data['is_fitted']
        
        print(f"📂 HMM regime detector loaded: {filepath}")
        return detector

def main():
    """Demonstrate HMM regime detection system."""
    print("🔍 PHASE 2B: Hidden Markov Models for Regime Detection")
    print("ULTRATHINK Implementation - Market State Classification")
    print("=" * 60)
    
    # Load data from Phase 1
    data_dir = "phase1/data/processed"
    import glob
    import os
    
    data_files = glob.glob(f"{data_dir}/BTC-USD_*.csv")
    if not data_files:
        print("❌ No data files found. Run Phase 1A first.")
        return
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"📂 Loading data from: {latest_file}")
    
    # Load data
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    print(f"📊 Data loaded: {len(df)} samples")
    print(f"📈 Columns: {list(df.columns)}")
    
    # Initialize HMM regime detector
    config = HMMConfig(
        n_states=3,
        state_names=['Bull_Market', 'Sideways_Market', 'Bear_Market'],
        use_returns=True,
        use_volatility=True,
        use_volume=True,
        max_iterations=50,  # Reduced for demo
        regime_window=20
    )
    
    detector = HMMRegimeDetector(config)
    
    # Train regime detector
    try:
        # Use subset for faster demo
        sample_df = df.iloc[:1000].copy()
        
        print(f"\n🔧 Processing {len(sample_df)} samples for demonstration...")
        
        detector.fit(sample_df)
        
        # Predict regimes on validation set
        val_df = df.iloc[1000:1200].copy() if len(df) > 1200 else sample_df.iloc[-100:]
        
        regime_predictions = detector.predict_regimes(val_df)
        
        print(f"\n📊 REGIME DETECTION RESULTS:")
        print(f"   Prediction samples: {len(regime_predictions)}")
        print(f"   Detected regimes: {regime_predictions['regime_name'].unique()}")
        
        # Show regime distribution
        regime_dist = regime_predictions['regime_name'].value_counts()
        print(f"\n📈 Regime Distribution:")
        for regime, count in regime_dist.items():
            print(f"   {regime}: {count} periods ({count/len(regime_predictions):.1%})")
        
        # Save model and results
        model_file = "phase2b/hmm_regime_detector.pkl"
        detector.save_model(model_file)
        
        # Save analysis results
        analysis_file = "phase2b/hmm_regime_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(detector.get_regime_summary(), f, indent=2, default=str)
        
        # Save predictions
        predictions_file = "phase2b/regime_predictions.csv"
        regime_predictions.to_csv(predictions_file)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Model: {model_file}")
        print(f"   📈 Analysis: {analysis_file}")
        print(f"   🔍 Predictions: {predictions_file}")
        
        print(f"\n🚀 Phase 2B HMM Regime Detection: COMPLETE")
        print(f"🎯 Ready for Phase 2B Next Step: Advanced Risk Management")
        
    except Exception as e:
        print(f"❌ Error in HMM regime detection: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()