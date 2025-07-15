"""UltraThink Crypto Trading System: Main integration script with advanced reasoning."""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import UltraThink components
from ultrathink import (
    UltraThinkReasoningEngine, 
    UltraThinkMarketAnalyzer, 
    UltraThinkStrategySelector,
    UltraThinkDecisionFramework
)
from ultrathink.decision_framework import DecisionType, DecisionPriority, MarketContext, TradingDecision
from ultrathink.strategy_selector import MarketRegime, StrategyRecommendation

# Import existing system components
from data.yfinance_fetcher import YFinanceCryptoFetcher
from features.ultra_feature_engineering import UltraFeatureEngine
from utils.config import load_config, get_default_config
from utils.checkpoint_manager import get_checkpoint_manager, start_auto_checkpoints
from utils.visualization import create_full_report


class UltraThinkTradingSystem:
    """Complete UltraThink-enhanced cryptocurrency trading system."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path) if config_path else get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize UltraThink components
        self.reasoning_engine = UltraThinkReasoningEngine()
        self.market_analyzer = UltraThinkMarketAnalyzer()
        self.strategy_selector = UltraThinkStrategySelector()
        self.decision_framework = UltraThinkDecisionFramework()
        
        # Initialize data components
        self.data_fetcher = None
        self.feature_engine = None
        
        # System state
        self.market_data = {}
        self.market_states = {}
        self.active_strategies = {}
        self.portfolio = {
            'total_value': 100000.0,  # Starting capital
            'available_capital': 100000.0,
            'positions': {},
            'trade_history': [],
            'current_risk': 0.0
        }
        
        # Start checkpoint system
        self.checkpoint_manager = start_auto_checkpoints(str(project_root))
        
        self.logger.info("UltraThink Trading System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ultrathink_trading.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    async def initialize_system(self):
        """Initialize all system components."""
        self.logger.info("Initializing UltraThink trading system...")
        
        # Initialize data fetcher
        if hasattr(self.config, 'data'):
            self.data_fetcher = YFinanceCryptoFetcher(self.config.data)
        else:
            # Create default data config
            from types import SimpleNamespace
            data_config = SimpleNamespace()
            data_config.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
            data_config.days = 365
            data_config.interval = '1h'
            self.data_fetcher = YFinanceCryptoFetcher(data_config)
        
        # Initialize feature engine
        self.feature_engine = UltraFeatureEngine()
        
        # Start decision framework
        await self.decision_framework.start_real_time_processing()
        
        self.logger.info("System initialization complete")
    
    async def run_ultrathink_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run complete UltraThink analysis for a symbol."""
        
        self.logger.info(f"Starting UltraThink analysis for {symbol}")
        
        try:
            # 1. Fetch and prepare data
            market_data = await self._fetch_market_data(symbol)
            if market_data is None or len(market_data) < 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # 2. Engineer features
            enhanced_data = self._engineer_features(market_data, symbol)
            
            # 3. Perform multi-dimensional market analysis
            market_state = self.market_analyzer.analyze_market(enhanced_data, symbol)
            self.market_states[symbol] = market_state
            
            # 4. Generate reasoning chain about market conditions
            reasoning_chain = self.reasoning_engine.reason_about_market(
                enhanced_data, 
                {'symbol': symbol, 'analysis_type': 'comprehensive'}
            )
            
            # 5. Select optimal strategy
            strategy_recommendation = self.strategy_selector.select_optimal_strategy(
                enhanced_data, symbol, {'portfolio_value': self.portfolio['total_value']}
            )
            
            # 6. Create market context for decision making
            market_context = MarketContext(
                timestamp=datetime.now(),
                market_state=market_state,
                active_positions=self.portfolio['positions'],
                portfolio_value=self.portfolio['total_value'],
                available_capital=self.portfolio['available_capital'],
                current_risk=self.portfolio['current_risk'],
                recent_decisions=[],
                market_regime=self._detect_market_regime(market_state),
                volatility_level=market_state.risk_level
            )
            
            # 7. Generate trading decision
            trading_decision = await self.decision_framework.make_decision(
                enhanced_data, symbol, DecisionType.ENTRY, market_context
            )
            
            # 8. Compile comprehensive analysis
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_analysis': {
                    'overall_sentiment': market_state.overall_sentiment,
                    'confidence': market_state.confidence_score,
                    'risk_level': market_state.risk_level,
                    'opportunity_score': market_state.opportunity_score,
                    'price': market_state.price,
                    'volume': market_state.volume
                },
                'reasoning_summary': {
                    'final_conclusion': reasoning_chain.final_conclusion,
                    'confidence': reasoning_chain.overall_confidence,
                    'reasoning_levels': len(set(node.level for node in reasoning_chain.nodes)),
                    'key_insights': [
                        {
                            'level': node.level.name,
                            'conclusion': node.conclusion,
                            'confidence': node.confidence
                        }
                        for node in reasoning_chain.nodes if node.confidence > 0.7
                    ]
                },
                'strategy_recommendation': {
                    'strategy_name': strategy_recommendation.strategy_config.name,
                    'strategy_type': strategy_recommendation.strategy_config.strategy_type.value,
                    'confidence': strategy_recommendation.confidence,
                    'expected_return': strategy_recommendation.expected_return,
                    'risk_score': strategy_recommendation.risk_score,
                    'market_fit': strategy_recommendation.market_fit_score,
                    'implementation_notes': strategy_recommendation.implementation_notes,
                    'risk_factors': strategy_recommendation.risk_factors
                },
                'trading_decision': {
                    'decision_id': trading_decision.decision_id,
                    'action': trading_decision.action,
                    'quantity': trading_decision.quantity,
                    'confidence': trading_decision.confidence,
                    'risk_score': trading_decision.risk_score,
                    'expected_outcome': trading_decision.expected_outcome,
                    'implementation_plan': trading_decision.implementation_plan,
                    'risk_mitigation': trading_decision.risk_mitigation
                },
                'dimensional_analysis': self.market_analyzer.get_analysis_summary(market_state)
            }
            
            self.logger.info(f"UltraThink analysis completed for {symbol}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in UltraThink analysis for {symbol}: {e}")
            raise
    
    async def _fetch_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for symbol."""
        try:
            # Convert crypto symbol format if needed
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"
            
            data_dict = self.data_fetcher.fetch_symbol_data(symbol)
            if not data_dict:
                return None
            
            # Get the actual data (assuming single symbol)
            data = list(data_dict.values())[0] if data_dict else None
            
            if data is not None and len(data) > 0:
                self.market_data[symbol] = data
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Engineer features for the data."""
        try:
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    self.logger.warning(f"Missing column {col} for {symbol}")
                    return data
            
            # Apply feature engineering
            enhanced_data = self.feature_engine.generate_features(data, symbol)
            
            self.logger.debug(f"Feature engineering completed for {symbol}: {enhanced_data.shape}")
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering for {symbol}: {e}")
            return data
    
    def _detect_market_regime(self, market_state) -> str:
        """Detect market regime from market state."""
        
        sentiment = market_state.overall_sentiment
        risk_level = market_state.risk_level
        confidence = market_state.confidence_score
        
        if confidence < 0.5:
            return "uncertain"
        elif risk_level in ['high', 'very_high']:
            return "volatile"
        elif sentiment == 'bullish' and confidence > 0.7:
            return "bull_trend"
        elif sentiment == 'bearish' and confidence > 0.7:
            return "bear_trend"
        else:
            return "sideways"
    
    async def run_portfolio_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Run portfolio-wide UltraThink analysis."""
        
        self.logger.info(f"Running portfolio analysis for {len(symbols)} symbols")
        
        portfolio_analysis = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': symbols,
            'individual_analyses': {},
            'portfolio_summary': {},
            'recommendations': [],
            'risk_assessment': {},
            'opportunity_ranking': []
        }
        
        # Analyze each symbol
        for symbol in symbols:
            try:
                analysis = await self.run_ultrathink_analysis(symbol)
                portfolio_analysis['individual_analyses'][symbol] = analysis
                
                # Add to opportunity ranking
                opportunity_score = analysis['market_analysis']['opportunity_score']
                confidence = analysis['market_analysis']['confidence']
                portfolio_analysis['opportunity_ranking'].append({
                    'symbol': symbol,
                    'opportunity_score': opportunity_score,
                    'confidence': confidence,
                    'combined_score': opportunity_score * confidence
                })
                
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                portfolio_analysis['individual_analyses'][symbol] = {'error': str(e)}
        
        # Sort opportunities
        portfolio_analysis['opportunity_ranking'].sort(
            key=lambda x: x['combined_score'], reverse=True
        )
        
        # Generate portfolio summary
        portfolio_analysis['portfolio_summary'] = self._generate_portfolio_summary(
            portfolio_analysis['individual_analyses']
        )
        
        # Generate portfolio recommendations
        portfolio_analysis['recommendations'] = self._generate_portfolio_recommendations(
            portfolio_analysis['individual_analyses'],
            portfolio_analysis['opportunity_ranking']
        )
        
        # Risk assessment
        portfolio_analysis['risk_assessment'] = self._assess_portfolio_risk(
            portfolio_analysis['individual_analyses']
        )
        
        self.logger.info("Portfolio analysis completed")
        return portfolio_analysis
    
    def _generate_portfolio_summary(self, individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio-level summary."""
        
        valid_analyses = {k: v for k, v in individual_analyses.items() if 'error' not in v}
        
        if not valid_analyses:
            return {'error': 'No valid analyses available'}
        
        # Aggregate metrics
        sentiments = [a['market_analysis']['overall_sentiment'] for a in valid_analyses.values()]
        confidences = [a['market_analysis']['confidence'] for a in valid_analyses.values()]
        risk_levels = [a['market_analysis']['risk_level'] for a in valid_analyses.values()]
        
        # Count sentiments
        bullish_count = sentiments.count('bullish')
        bearish_count = sentiments.count('bearish')
        neutral_count = sentiments.count('neutral')
        
        # Determine overall portfolio sentiment
        if bullish_count > bearish_count * 1.5:
            portfolio_sentiment = 'bullish'
        elif bearish_count > bullish_count * 1.5:
            portfolio_sentiment = 'bearish'
        else:
            portfolio_sentiment = 'mixed'
        
        # Risk distribution
        risk_distribution = {
            'low': risk_levels.count('low'),
            'medium': risk_levels.count('medium'),
            'high': risk_levels.count('high'),
            'very_high': risk_levels.count('very_high')
        }
        
        return {
            'portfolio_sentiment': portfolio_sentiment,
            'average_confidence': np.mean(confidences),
            'sentiment_distribution': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'risk_distribution': risk_distribution,
            'total_assets_analyzed': len(valid_analyses),
            'high_confidence_signals': len([c for c in confidences if c > 0.7]),
            'consensus_strength': max(bullish_count, bearish_count, neutral_count) / len(sentiments)
        }
    
    def _generate_portfolio_recommendations(self, individual_analyses: Dict[str, Any],
                                          opportunity_ranking: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate portfolio-level recommendations."""
        
        recommendations = []
        
        # Top opportunities
        top_opportunities = opportunity_ranking[:3]  # Top 3
        for opp in top_opportunities:
            symbol = opp['symbol']
            if symbol in individual_analyses and 'error' not in individual_analyses[symbol]:
                analysis = individual_analyses[symbol]
                recommendations.append({
                    'type': 'opportunity',
                    'symbol': symbol,
                    'reason': f"High opportunity score ({opp['opportunity_score']:.2%}) with strong confidence",
                    'action': analysis['trading_decision']['action'],
                    'priority': 'high' if opp['combined_score'] > 0.6 else 'medium'
                })
        
        # Risk warnings
        for symbol, analysis in individual_analyses.items():
            if 'error' in analysis:
                continue
            
            risk_level = analysis['market_analysis']['risk_level']
            if risk_level in ['high', 'very_high']:
                recommendations.append({
                    'type': 'risk_warning',
                    'symbol': symbol,
                    'reason': f"Elevated risk level: {risk_level}",
                    'action': 'reduce_exposure',
                    'priority': 'high' if risk_level == 'very_high' else 'medium'
                })
        
        # Diversification recommendations
        sentiments = [a['market_analysis']['overall_sentiment'] 
                     for a in individual_analyses.values() if 'error' not in a]
        
        if len(set(sentiments)) == 1 and len(sentiments) > 1:
            recommendations.append({
                'type': 'diversification',
                'symbol': 'portfolio',
                'reason': 'All positions show same sentiment - consider diversification',
                'action': 'diversify',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _assess_portfolio_risk(self, individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk."""
        
        valid_analyses = {k: v for k, v in individual_analyses.items() if 'error' not in v}
        
        if not valid_analyses:
            return {'error': 'No valid analyses for risk assessment'}
        
        # Risk scores
        risk_scores = [a['trading_decision']['risk_score'] for a in valid_analyses.values()]
        market_risks = [a['market_analysis']['risk_level'] for a in valid_analyses.values()]
        
        # Portfolio risk metrics
        avg_risk_score = np.mean(risk_scores)
        max_risk_score = np.max(risk_scores)
        
        # Risk concentration
        high_risk_assets = len([r for r in market_risks if r in ['high', 'very_high']])
        risk_concentration = high_risk_assets / len(valid_analyses)
        
        # Overall risk level
        if max_risk_score > 0.8 or risk_concentration > 0.5:
            overall_risk = 'high'
        elif avg_risk_score > 0.6 or risk_concentration > 0.3:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'overall_risk_level': overall_risk,
            'average_risk_score': avg_risk_score,
            'maximum_risk_score': max_risk_score,
            'risk_concentration': risk_concentration,
            'high_risk_asset_count': high_risk_assets,
            'total_assets': len(valid_analyses),
            'risk_diversification': 1 - risk_concentration,
            'recommendations': self._generate_risk_recommendations(overall_risk, risk_concentration)
        }
    
    def _generate_risk_recommendations(self, overall_risk: str, concentration: float) -> List[str]:
        """Generate risk management recommendations."""
        
        recommendations = []
        
        if overall_risk == 'high':
            recommendations.append("Consider reducing overall position sizes")
            recommendations.append("Implement stricter stop-loss levels")
            recommendations.append("Increase cash allocation for defensive positioning")
        
        if concentration > 0.4:
            recommendations.append("High risk concentration detected - diversify across risk levels")
            recommendations.append("Consider adding low-risk assets to portfolio")
        
        if overall_risk in ['medium', 'high']:
            recommendations.append("Monitor positions more frequently")
            recommendations.append("Be prepared to implement emergency stops if needed")
        
        return recommendations
    
    async def save_analysis_results(self, analysis_results: Dict[str, Any], 
                                  output_dir: str = None) -> Dict[str, str]:
        """Save analysis results and generate reports."""
        
        if output_dir is None:
            output_dir = f"ultrathink_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save JSON results
        json_file = output_path / "ultrathink_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        saved_files['analysis_json'] = str(json_file)
        
        # Save reasoning summaries
        if 'individual_analyses' in analysis_results:
            reasoning_file = output_path / "reasoning_summaries.json"
            reasoning_data = {}
            
            for symbol, analysis in analysis_results['individual_analyses'].items():
                if 'error' not in analysis:
                    reasoning_data[symbol] = analysis['reasoning_summary']
            
            with open(reasoning_file, 'w') as f:
                json.dump(reasoning_data, f, indent=2)
            saved_files['reasoning_summaries'] = str(reasoning_file)
        
        # Save recommendations
        if 'recommendations' in analysis_results:
            rec_file = output_path / "recommendations.json"
            with open(rec_file, 'w') as f:
                json.dump(analysis_results['recommendations'], f, indent=2)
            saved_files['recommendations'] = str(rec_file)
        
        # Create emergency checkpoint
        checkpoint_path = self.checkpoint_manager.create_checkpoint(
            f"UltraThink analysis completed: {datetime.now()}"
        )
        saved_files['emergency_checkpoint'] = checkpoint_path
        
        self.logger.info(f"Analysis results saved to {output_dir}")
        return saved_files
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        self.logger.info("Shutting down UltraThink trading system...")
        
        # Stop decision framework
        await self.decision_framework.stop_real_time_processing()
        
        # Stop checkpoint manager
        if hasattr(self.checkpoint_manager, 'stop_auto_checkpoint'):
            self.checkpoint_manager.stop_auto_checkpoint()
        
        self.logger.info("System shutdown complete")


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='UltraThink Crypto Trading System')
    
    parser.add_argument('--symbols', nargs='+', default=['BTC-USD', 'ETH-USD', 'SOL-USD'],
                       help='Cryptocurrency symbols to analyze')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--single-symbol', type=str, default=None,
                       help='Analyze single symbol only')
    parser.add_argument('--portfolio-mode', action='store_true',
                       help='Run portfolio-wide analysis')
    parser.add_argument('--save-results', action='store_true', default=True,
                       help='Save analysis results to files')
    
    args = parser.parse_args()
    
    # Initialize system
    system = UltraThinkTradingSystem(args.config)
    
    try:
        await system.initialize_system()
        
        if args.single_symbol:
            # Single symbol analysis
            print(f"\n🧠 Running UltraThink analysis for {args.single_symbol}...")
            analysis = await system.run_ultrathink_analysis(args.single_symbol)
            
            # Print summary
            print(f"\n📊 Analysis Summary for {args.single_symbol}:")
            print(f"   Market Sentiment: {analysis['market_analysis']['overall_sentiment'].upper()}")
            print(f"   Confidence: {analysis['market_analysis']['confidence']:.1%}")
            print(f"   Risk Level: {analysis['market_analysis']['risk_level'].upper()}")
            print(f"   Opportunity Score: {analysis['market_analysis']['opportunity_score']:.1%}")
            print(f"   Recommended Action: {analysis['trading_decision']['action'].upper()}")
            print(f"   Strategy: {analysis['strategy_recommendation']['strategy_name']}")
            
            if args.save_results:
                saved_files = await system.save_analysis_results(analysis, args.output)
                print(f"\n💾 Results saved to: {list(saved_files.values())[0]}")
        
        elif args.portfolio_mode:
            # Portfolio analysis
            print(f"\n🧠 Running UltraThink portfolio analysis for {len(args.symbols)} symbols...")
            portfolio_analysis = await system.run_portfolio_analysis(args.symbols)
            
            # Print portfolio summary
            summary = portfolio_analysis['portfolio_summary']
            print(f"\n📊 Portfolio Analysis Summary:")
            print(f"   Overall Sentiment: {summary['portfolio_sentiment'].upper()}")
            print(f"   Average Confidence: {summary['average_confidence']:.1%}")
            print(f"   Assets Analyzed: {summary['total_assets_analyzed']}")
            print(f"   High Confidence Signals: {summary['high_confidence_signals']}")
            print(f"   Consensus Strength: {summary['consensus_strength']:.1%}")
            
            # Print top opportunities
            print(f"\n🎯 Top Opportunities:")
            for i, opp in enumerate(portfolio_analysis['opportunity_ranking'][:3], 1):
                print(f"   {i}. {opp['symbol']}: {opp['combined_score']:.1%} combined score")
            
            # Print recommendations
            print(f"\n💡 Key Recommendations:")
            for rec in portfolio_analysis['recommendations'][:5]:
                print(f"   • {rec['type'].title()}: {rec['reason']}")
            
            if args.save_results:
                saved_files = await system.save_analysis_results(portfolio_analysis, args.output)
                print(f"\n💾 Results saved to: {list(saved_files.values())[0]}")
        
        else:
            # Individual symbol analyses
            print(f"\n🧠 Running UltraThink analysis for {len(args.symbols)} symbols...")
            
            results = {}
            for symbol in args.symbols:
                print(f"\nAnalyzing {symbol}...")
                try:
                    analysis = await system.run_ultrathink_analysis(symbol)
                    results[symbol] = analysis
                    
                    # Quick summary
                    sentiment = analysis['market_analysis']['overall_sentiment']
                    confidence = analysis['market_analysis']['confidence']
                    action = analysis['trading_decision']['action']
                    print(f"   {symbol}: {sentiment.upper()} ({confidence:.1%} conf) → {action.upper()}")
                    
                except Exception as e:
                    print(f"   {symbol}: ERROR - {e}")
                    results[symbol] = {'error': str(e)}
            
            if args.save_results:
                saved_files = await system.save_analysis_results(
                    {'individual_analyses': results}, args.output
                )
                print(f"\n💾 Results saved to: {list(saved_files.values())[0]}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await system.shutdown()
        print("\n✅ UltraThink system shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())