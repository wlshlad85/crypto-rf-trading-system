# Data Pipeline Context

## Overview
Handles real-time and historical market data ingestion, validation, and distribution for the institutional-grade cryptocurrency trading system. Provides 99.5% validated market data with sub-second latency for trading decisions.

## Critical Files
- `data/data_fetcher.py` - Primary data interface with caching and validation
- `data/yfinance_fetcher.py` - Yahoo Finance integration with rate limiting
- `data/minute_data_manager.py` - High-frequency data handling and storage
- `data/multi_timeframe_fetcher.py` - Multi-resolution data pipeline
- `phase1/enhanced_data_collector.py` - 99.5% quality validation framework

## Key Classes and Functions

### DataFetcher
- **Purpose**: Primary interface for market data retrieval with caching
- **Key Methods**:
  - `fetch_ohlcv_data()` - Get market data with automatic validation
  - `validate_data_quality()` - Ensure 99.5% data integrity threshold
  - `cache_data()` - Intelligent caching for performance optimization
  - `handle_missing_data()` - Forward-fill with configurable limits

### MinuteDataManager
- **Purpose**: Real-time minute-by-minute data processing
- **Key Methods**:
  - `stream_realtime_data()` - WebSocket data streaming with reconnection
  - `process_tick_data()` - Convert ticks to OHLCV format
  - `validate_realtime()` - Real-time anomaly detection
- **Performance**: Handles 10,000+ ticks/second

### EnhancedDataCollector
- **Purpose**: Institutional-grade data validation and quality control
- **Key Methods**:
  - `collect_and_validate()` - Multi-stage validation pipeline
  - `generate_quality_report()` - Comprehensive data quality metrics
  - `handle_data_gaps()` - Smart gap detection and filling
- **Quality Target**: 99.5% validation threshold

## Dependencies
- **Internal**: utils/config.py, utils/checkpoint_manager.py
- **External**: pandas, numpy, yfinance, requests, websocket-client

## Configuration
- **Environment Variables**: 
  - `DATA_CACHE_SIZE` - Cache size limit (default: 1GB)
  - `DATA_VALIDATION_THRESHOLD` - Quality threshold (default: 99.5%)
  - `REALTIME_BUFFER_SIZE` - Streaming buffer size (default: 10000)
- **Config Files**: configs/config.json - data source configurations
- **Constants**: 
  - `MAX_MISSING_PERCENTAGE` = 0.5%
  - `CACHE_EXPIRY_MINUTES` = 15
  - `RECONNECTION_ATTEMPTS` = 5

## Common Patterns

### Data Fetching Pattern
```python
# Standard data fetching with validation
fetcher = DataFetcher()
data = fetcher.fetch_ohlcv_data(
    symbol='BTC-USD',
    start_date='2024-01-01',
    validate=True,
    cache=True
)
```

### Real-time Streaming Pattern
```python
# Real-time data streaming setup
manager = MinuteDataManager()
manager.start_stream('BTC-USD', callback=process_tick)
```

### Quality Validation Pattern
```python
# Data quality validation
collector = EnhancedDataCollector()
quality_report = collector.validate_dataset(data)
if quality_report['quality_score'] >= 0.995:
    proceed_with_trading(data)
```

## Performance Considerations
- **Memory Usage**: Implement rolling buffers for real-time data (max 100MB)
- **Latency Requirements**: < 100ms for data retrieval, < 10ms for validation
- **Cache Strategy**: LRU cache with intelligent prefetching for common symbols
- **Network Optimization**: Connection pooling and compression for API calls

## Integration Points
- **Phase 1**: Feeds validated data to walk_forward_engine.py
- **Phase 2**: Provides features to advanced_technical_indicators.py
- **Phase 2B**: Supplies data to ensemble_meta_learning.py and hmm_regime_detection.py
- **Live Trading**: Real-time feed to enhanced_paper_trader_24h.py
- **Risk Management**: Data validation alerts to advanced_risk_management.py

## Current Live Status
- **Active Session**: enhanced_paper_trader_24h.py receiving real-time BTC data
- **Data Quality**: Maintaining 99.5%+ validation threshold
- **Latency**: < 50ms average data retrieval
- **Cache Hit Rate**: 85%+ for frequently accessed symbols

## Testing
- **Test Files**: `tests/test_data_pipeline.py`
- **Key Test Scenarios**:
  - Network disconnection recovery
  - Data quality validation edge cases
  - Real-time streaming performance under load
  - Cache consistency and expiration
  - Missing data handling accuracy

## Known Issues & TODOs
- [ ] Implement backup data sources for redundancy
- [ ] Add data compression for long-term storage
- [ ] Optimize memory usage for 24/7 streaming
- [ ] Add support for additional cryptocurrency exchanges
- [ ] Implement predictive caching based on trading patterns

## Error Handling
- **Network Issues**: Automatic retry with exponential backoff
- **Invalid Data**: Quarantine and alert system
- **API Rate Limits**: Smart throttling and queue management
- **Memory Pressure**: Automatic cache cleanup and garbage collection

## Monitoring & Alerts
- **Quality Drops**: Alert if validation score < 99.5%
- **Latency Spikes**: Alert if data retrieval > 200ms
- **Connection Issues**: Alert on stream disconnection > 30s
- **Cache Performance**: Monitor hit rates and optimize accordingly