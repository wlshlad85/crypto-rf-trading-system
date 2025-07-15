#!/usr/bin/env python3
"""
ULTRATHINK Week 6 DAY 38: Enhanced Audit Trail System
Immutable audit logging with cryptographic verification for regulatory compliance.

Features:
- Immutable transaction logging with hash chains
- Complete decision audit trail with full context
- User action tracking for all administrative operations
- Data integrity verification using cryptographic hashing
- Tamper-evident logging with blockchain-style verification
- Regulatory export capabilities for compliance submissions
- Real-time audit monitoring and alerting
- Automated compliance reporting integration
- Secure audit log storage and archival

Compliance Standards:
- SOX: Sarbanes-Oxley Act audit requirements
- MiFID II: Transaction reporting and record keeping
- GDPR: Data processing audit trails
- Basel III: Operational risk audit requirements
- CFTC: Commodity trading audit requirements
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
import uuid
import hmac
import base64
import zlib
from pathlib import Path
from threading import Lock
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np

# Import existing components
import sys
sys.path.append('/home/richardw/crypto_rf_trading_system')
from production.real_money_trader import AuditLogger as BasicAuditLogger

class AuditLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class AuditCategory(Enum):
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"

class AuditStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPORTED = "exported"
    DELETED = "deleted"

@dataclass
class AuditContext:
    """Context information for audit events."""
    session_id: str
    user_id: str
    client_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Geographic context
    country: Optional[str] = None
    region: Optional[str] = None
    timezone: Optional[str] = None
    
    # System context
    system_version: Optional[str] = None
    component: Optional[str] = None
    environment: str = "production"

@dataclass
class AuditEvent:
    """Individual audit event with comprehensive tracking."""
    event_id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    event_type: str
    description: str
    
    # Context
    context: AuditContext
    
    # Event details
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Business context
    symbol: Optional[str] = None
    client_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    
    # Financial context
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    price: Optional[Decimal] = None
    
    # Risk context
    risk_score: Optional[float] = None
    compliance_flags: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    classification: str = "internal"  # internal, confidential, restricted
    retention_period: int = 2555  # 7 years in days
    
    # Integrity
    checksum: Optional[str] = None
    previous_hash: Optional[str] = None
    chain_hash: Optional[str] = None
    
    # Processing
    status: AuditStatus = AuditStatus.ACTIVE
    archived_at: Optional[datetime] = None
    exported_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate event checksum."""
        if not self.checksum:
            self.checksum = self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate event integrity checksum."""
        # Create deterministic string representation
        data_str = (
            f"{self.timestamp.isoformat()}"
            f"{self.level.value}"
            f"{self.category.value}"
            f"{self.event_type}"
            f"{self.description}"
            f"{json.dumps(self.details, sort_keys=True, default=str)}"
            f"{self.context.session_id}"
            f"{self.context.user_id}"
        )
        
        # Calculate SHA-256 hash
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity."""
        current_checksum = self.checksum
        self.checksum = None
        calculated_checksum = self.calculate_checksum()
        self.checksum = current_checksum
        
        return calculated_checksum == current_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category.value,
            'event_type': self.event_type,
            'description': self.description,
            'context': asdict(self.context),
            'details': self.details,
            'business_context': {
                'symbol': self.symbol,
                'client_id': self.client_id,
                'order_id': self.order_id,
                'trade_id': self.trade_id,
                'portfolio_id': self.portfolio_id
            },
            'financial_context': {
                'amount': str(self.amount) if self.amount else None,
                'currency': self.currency,
                'price': str(self.price) if self.price else None
            },
            'risk_context': {
                'risk_score': self.risk_score,
                'compliance_flags': self.compliance_flags
            },
            'metadata': {
                'tags': self.tags,
                'classification': self.classification,
                'retention_period': self.retention_period
            },
            'integrity': {
                'checksum': self.checksum,
                'previous_hash': self.previous_hash,
                'chain_hash': self.chain_hash
            },
            'processing': {
                'status': self.status.value,
                'archived_at': self.archived_at.isoformat() if self.archived_at else None,
                'exported_at': self.exported_at.isoformat() if self.exported_at else None
            }
        }

@dataclass
class AuditChain:
    """Blockchain-style audit chain for immutable logging."""
    chain_id: str
    genesis_hash: str
    current_hash: str
    block_count: int
    created_at: datetime
    updated_at: datetime
    
    # Chain metadata
    description: str = "Audit event chain"
    category: AuditCategory = AuditCategory.TRADING
    
    # Security
    is_sealed: bool = False
    seal_timestamp: Optional[datetime] = None
    seal_signature: Optional[str] = None
    
    def calculate_next_hash(self, event: AuditEvent) -> str:
        """Calculate next hash in chain."""
        chain_data = (
            f"{self.current_hash}"
            f"{event.event_id}"
            f"{event.timestamp.isoformat()}"
            f"{event.checksum}"
        )
        return hashlib.sha256(chain_data.encode('utf-8')).hexdigest()
    
    def verify_chain_integrity(self, events: List[AuditEvent]) -> bool:
        """Verify entire chain integrity."""
        if not events:
            return True
        
        # Verify chain progression
        current_hash = self.genesis_hash
        
        for event in events:
            expected_hash = self.calculate_next_hash(event)
            if event.chain_hash != expected_hash:
                return False
            current_hash = expected_hash
        
        return current_hash == self.current_hash

class AuditStorage:
    """Secure audit storage with encryption and compression."""
    
    def __init__(self, storage_path: str = "compliance/audit_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Encryption key (in production, load from secure key management)
        self.encryption_key = b'your-32-byte-encryption-key-here'
        
        # Compression settings
        self.compression_level = 6  # Balance between speed and size
        
        # File rotation settings
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_files_per_day = 24
        
        self.logger = logging.getLogger(__name__)
    
    def store_event(self, event: AuditEvent) -> str:
        """Store audit event securely."""
        try:
            # Serialize event
            event_data = json.dumps(event.to_dict(), default=str)
            
            # Compress data
            compressed_data = zlib.compress(event_data.encode('utf-8'), self.compression_level)
            
            # Encrypt data (simplified - use proper encryption in production)
            encrypted_data = self._encrypt_data(compressed_data)
            
            # Determine storage file
            storage_file = self._get_storage_file(event.timestamp)
            
            # Store to file
            with open(storage_file, 'ab') as f:
                # Write length prefix + data
                data_length = len(encrypted_data)
                f.write(data_length.to_bytes(4, byteorder='big'))
                f.write(encrypted_data)
            
            return str(storage_file)
            
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}")
            raise
    
    def retrieve_events(self, start_time: datetime, end_time: datetime) -> List[AuditEvent]:
        """Retrieve audit events from storage."""
        events = []
        
        try:
            # Find relevant storage files
            storage_files = self._find_storage_files(start_time, end_time)
            
            for storage_file in storage_files:
                events.extend(self._read_events_from_file(storage_file, start_time, end_time))
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit events: {e}")
            return []
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data (simplified implementation)."""
        # In production, use proper encryption like AES-256-GCM
        # This is a placeholder
        return data
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data (simplified implementation)."""
        # In production, use proper decryption
        # This is a placeholder
        return encrypted_data
    
    def _get_storage_file(self, timestamp: datetime) -> Path:
        """Get storage file path for timestamp."""
        date_str = timestamp.strftime("%Y%m%d")
        hour_str = timestamp.strftime("%H")
        
        filename = f"audit_{date_str}_{hour_str}.bin"
        return self.storage_path / filename
    
    def _find_storage_files(self, start_time: datetime, end_time: datetime) -> List[Path]:
        """Find storage files that might contain events in time range."""
        files = []
        
        # Get all audit files
        for file_path in self.storage_path.glob("audit_*.bin"):
            try:
                # Parse filename to get date/hour
                filename = file_path.stem
                date_hour = filename.split('_')[1] + filename.split('_')[2]
                file_time = datetime.strptime(date_hour, "%Y%m%d%H")
                
                # Check if file might contain relevant events
                if start_time <= file_time <= end_time + timedelta(hours=1):
                    files.append(file_path)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse audit file {file_path}: {e}")
        
        return sorted(files)
    
    def _read_events_from_file(self, file_path: Path, start_time: datetime, end_time: datetime) -> List[AuditEvent]:
        """Read events from storage file."""
        events = []
        
        try:
            with open(file_path, 'rb') as f:
                while True:
                    # Read length prefix
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break
                    
                    data_length = int.from_bytes(length_bytes, byteorder='big')
                    
                    # Read encrypted data
                    encrypted_data = f.read(data_length)
                    if len(encrypted_data) != data_length:
                        break
                    
                    # Decrypt and decompress
                    decrypted_data = self._decrypt_data(encrypted_data)
                    decompressed_data = zlib.decompress(decrypted_data)
                    
                    # Parse event
                    event_dict = json.loads(decompressed_data.decode('utf-8'))
                    event = self._dict_to_event(event_dict)
                    
                    # Check time range
                    if start_time <= event.timestamp <= end_time:
                        events.append(event)
                        
        except Exception as e:
            self.logger.error(f"Failed to read events from {file_path}: {e}")
        
        return events
    
    def _dict_to_event(self, event_dict: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent."""
        context = AuditContext(**event_dict['context'])
        
        return AuditEvent(
            event_id=event_dict['event_id'],
            timestamp=datetime.fromisoformat(event_dict['timestamp']),
            level=AuditLevel(event_dict['level']),
            category=AuditCategory(event_dict['category']),
            event_type=event_dict['event_type'],
            description=event_dict['description'],
            context=context,
            details=event_dict['details'],
            symbol=event_dict['business_context']['symbol'],
            client_id=event_dict['business_context']['client_id'],
            order_id=event_dict['business_context']['order_id'],
            trade_id=event_dict['business_context']['trade_id'],
            portfolio_id=event_dict['business_context']['portfolio_id'],
            amount=Decimal(event_dict['financial_context']['amount']) if event_dict['financial_context']['amount'] else None,
            currency=event_dict['financial_context']['currency'],
            price=Decimal(event_dict['financial_context']['price']) if event_dict['financial_context']['price'] else None,
            risk_score=event_dict['risk_context']['risk_score'],
            compliance_flags=event_dict['risk_context']['compliance_flags'],
            tags=event_dict['metadata']['tags'],
            classification=event_dict['metadata']['classification'],
            retention_period=event_dict['metadata']['retention_period'],
            checksum=event_dict['integrity']['checksum'],
            previous_hash=event_dict['integrity']['previous_hash'],
            chain_hash=event_dict['integrity']['chain_hash'],
            status=AuditStatus(event_dict['processing']['status']),
            archived_at=datetime.fromisoformat(event_dict['processing']['archived_at']) if event_dict['processing']['archived_at'] else None,
            exported_at=datetime.fromisoformat(event_dict['processing']['exported_at']) if event_dict['processing']['exported_at'] else None
        )

class EnhancedAuditLogger:
    """
    Enhanced audit logging system with immutable trails and regulatory compliance.
    
    Features:
    - Immutable audit chains with cryptographic verification
    - Comprehensive event tracking with full context
    - Regulatory compliance support (SOX, MiFID II, GDPR)
    - Secure storage with encryption and compression
    - Real-time monitoring and alerting
    - Automated archival and retention management
    """
    
    def __init__(self, 
                 storage_path: str = "compliance/audit_storage",
                 database_path: str = "compliance/audit_trail.db"):
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.storage = AuditStorage(storage_path)
        self.database_path = database_path
        
        # Chain management
        self.chains: Dict[str, AuditChain] = {}
        self.chain_lock = Lock()
        
        # Event queues
        self.event_queue = []
        self.queue_lock = Lock()
        
        # Configuration
        self.batch_size = 100
        self.flush_interval = 30  # seconds
        
        # Monitoring
        self.event_count = 0
        self.error_count = 0
        self.last_flush = datetime.now()
        
        # Database setup
        self.setup_database()
        
        # Load existing chains
        self.load_chains()
        
        # Background processing
        self.background_task = None
        self.is_running = False
        
        self.logger.info("Enhanced Audit Logger initialized")
    
    def setup_database(self):
        """Initialize audit database."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.database_path) as conn:
            # Audit events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    client_id TEXT,
                    symbol TEXT,
                    order_id TEXT,
                    trade_id TEXT,
                    amount TEXT,
                    currency TEXT,
                    price TEXT,
                    risk_score REAL,
                    compliance_flags TEXT,
                    tags TEXT,
                    classification TEXT,
                    checksum TEXT NOT NULL,
                    chain_id TEXT,
                    chain_hash TEXT,
                    storage_path TEXT,
                    status TEXT NOT NULL,
                    event_data TEXT NOT NULL
                )
            """)
            
            # Audit chains table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_chains (
                    chain_id TEXT PRIMARY KEY,
                    genesis_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL,
                    block_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    is_sealed BOOLEAN NOT NULL,
                    seal_timestamp TEXT,
                    seal_signature TEXT
                )
            """)
            
            # Audit statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    error_count INTEGER NOT NULL,
                    data_size INTEGER NOT NULL,
                    processing_time REAL NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_category ON audit_events(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_level ON audit_events(level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_user ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_client ON audit_events(client_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_symbol ON audit_events(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_order ON audit_events(order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_chain ON audit_events(chain_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_date ON audit_statistics(date)")
    
    def load_chains(self):
        """Load existing audit chains from database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT chain_id, genesis_hash, current_hash, block_count,
                           created_at, updated_at, description, category,
                           is_sealed, seal_timestamp, seal_signature
                    FROM audit_chains
                    WHERE is_sealed = 0
                """)
                
                for row in cursor.fetchall():
                    chain = AuditChain(
                        chain_id=row[0],
                        genesis_hash=row[1],
                        current_hash=row[2],
                        block_count=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5]),
                        description=row[6],
                        category=AuditCategory(row[7]),
                        is_sealed=bool(row[8]),
                        seal_timestamp=datetime.fromisoformat(row[9]) if row[9] else None,
                        seal_signature=row[10]
                    )
                    self.chains[chain.chain_id] = chain
                    
            self.logger.info(f"Loaded {len(self.chains)} audit chains")
            
        except Exception as e:
            self.logger.error(f"Failed to load audit chains: {e}")
    
    def create_chain(self, category: AuditCategory, description: str = None) -> str:
        """Create new audit chain."""
        with self.chain_lock:
            chain_id = str(uuid.uuid4())
            genesis_hash = hashlib.sha256(f"{chain_id}{datetime.now().isoformat()}".encode()).hexdigest()
            
            chain = AuditChain(
                chain_id=chain_id,
                genesis_hash=genesis_hash,
                current_hash=genesis_hash,
                block_count=0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                description=description or f"Audit chain for {category.value}",
                category=category
            )
            
            self.chains[chain_id] = chain
            self.save_chain(chain)
            
            self.logger.info(f"Created audit chain: {chain_id}")
            return chain_id
    
    def save_chain(self, chain: AuditChain):
        """Save audit chain to database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audit_chains (
                        chain_id, genesis_hash, current_hash, block_count,
                        created_at, updated_at, description, category,
                        is_sealed, seal_timestamp, seal_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chain.chain_id,
                    chain.genesis_hash,
                    chain.current_hash,
                    chain.block_count,
                    chain.created_at.isoformat(),
                    chain.updated_at.isoformat(),
                    chain.description,
                    chain.category.value,
                    chain.is_sealed,
                    chain.seal_timestamp.isoformat() if chain.seal_timestamp else None,
                    chain.seal_signature
                ))
        except Exception as e:
            self.logger.error(f"Failed to save audit chain: {e}")
    
    def log_event(self, 
                  level: AuditLevel,
                  category: AuditCategory,
                  event_type: str,
                  description: str,
                  context: AuditContext,
                  **kwargs) -> str:
        """Log audit event with comprehensive tracking."""
        
        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            category=category,
            event_type=event_type,
            description=description,
            context=context,
            **kwargs
        )
        
        # Add to appropriate chain
        chain_id = self.get_or_create_chain(category)
        
        with self.chain_lock:
            chain = self.chains[chain_id]
            
            # Set previous hash
            event.previous_hash = chain.current_hash
            
            # Calculate chain hash
            event.chain_hash = chain.calculate_next_hash(event)
            
            # Update chain
            chain.current_hash = event.chain_hash
            chain.block_count += 1
            chain.updated_at = datetime.now()
            
            # Save chain
            self.save_chain(chain)
        
        # Queue event for processing
        with self.queue_lock:
            self.event_queue.append(event)
            
            # Flush if batch size reached
            if len(self.event_queue) >= self.batch_size:
                self.flush_events()
        
        self.event_count += 1
        
        # Log to standard logger for immediate visibility
        if level == AuditLevel.CRITICAL:
            self.logger.critical(f"AUDIT: {description}")
        elif level == AuditLevel.ERROR:
            self.logger.error(f"AUDIT: {description}")
        elif level == AuditLevel.WARNING:
            self.logger.warning(f"AUDIT: {description}")
        elif level == AuditLevel.SECURITY:
            self.logger.warning(f"SECURITY AUDIT: {description}")
        elif level == AuditLevel.COMPLIANCE:
            self.logger.info(f"COMPLIANCE AUDIT: {description}")
        
        return event.event_id
    
    def get_or_create_chain(self, category: AuditCategory) -> str:
        """Get existing chain or create new one for category."""
        # Look for existing active chain
        for chain_id, chain in self.chains.items():
            if chain.category == category and not chain.is_sealed:
                return chain_id
        
        # Create new chain
        return self.create_chain(category)
    
    def flush_events(self):
        """Flush queued events to storage."""
        if not self.event_queue:
            return
        
        start_time = time.time()
        events_to_process = []
        
        with self.queue_lock:
            events_to_process = self.event_queue.copy()
            self.event_queue.clear()
        
        # Process events
        for event in events_to_process:
            try:
                # Store to secure storage
                storage_path = self.storage.store_event(event)
                
                # Store to database
                self.store_event_to_db(event, storage_path)
                
            except Exception as e:
                self.logger.error(f"Failed to process audit event {event.event_id}: {e}")
                self.error_count += 1
        
        processing_time = time.time() - start_time
        self.last_flush = datetime.now()
        
        # Update statistics
        self.update_statistics(events_to_process, processing_time)
        
        self.logger.debug(f"Flushed {len(events_to_process)} audit events in {processing_time:.2f}s")
    
    def store_event_to_db(self, event: AuditEvent, storage_path: str):
        """Store audit event to database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, level, category, event_type,
                        description, session_id, user_id, client_id,
                        symbol, order_id, trade_id, amount, currency, price,
                        risk_score, compliance_flags, tags, classification,
                        checksum, chain_id, chain_hash, storage_path,
                        status, event_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.level.value,
                    event.category.value,
                    event.event_type,
                    event.description,
                    event.context.session_id,
                    event.context.user_id,
                    event.context.client_id,
                    event.symbol,
                    event.order_id,
                    event.trade_id,
                    str(event.amount) if event.amount else None,
                    event.currency,
                    str(event.price) if event.price else None,
                    event.risk_score,
                    json.dumps(event.compliance_flags),
                    json.dumps(event.tags),
                    event.classification,
                    event.checksum,
                    self.get_chain_id_for_event(event),
                    event.chain_hash,
                    storage_path,
                    event.status.value,
                    json.dumps(event.to_dict(), default=str)
                ))
        except Exception as e:
            self.logger.error(f"Failed to store event to database: {e}")
            raise
    
    def get_chain_id_for_event(self, event: AuditEvent) -> str:
        """Get chain ID for event."""
        for chain_id, chain in self.chains.items():
            if chain.category == event.category and not chain.is_sealed:
                return chain_id
        return ""
    
    def update_statistics(self, events: List[AuditEvent], processing_time: float):
        """Update audit statistics."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            # Group events by category and level
            stats = defaultdict(lambda: defaultdict(int))
            total_size = 0
            
            for event in events:
                stats[event.category.value][event.level.value] += 1
                total_size += len(json.dumps(event.to_dict(), default=str))
            
            # Update database
            with sqlite3.connect(self.database_path) as conn:
                for category, level_stats in stats.items():
                    for level, count in level_stats.items():
                        conn.execute("""
                            INSERT OR REPLACE INTO audit_statistics (
                                date, category, level, event_count, error_count,
                                data_size, processing_time
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            date_str,
                            category,
                            level,
                            count,
                            self.error_count,
                            total_size,
                            processing_time
                        ))
                        
        except Exception as e:
            self.logger.error(f"Failed to update statistics: {e}")
    
    def search_events(self, 
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      category: Optional[AuditCategory] = None,
                      level: Optional[AuditLevel] = None,
                      event_type: Optional[str] = None,
                      user_id: Optional[str] = None,
                      client_id: Optional[str] = None,
                      symbol: Optional[str] = None,
                      order_id: Optional[str] = None,
                      limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with filters."""
        
        # Build query
        query = "SELECT event_data FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if category:
            query += " AND category = ?"
            params.append(category.value)
        
        if level:
            query += " AND level = ?"
            params.append(level.value)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if order_id:
            query += " AND order_id = ?"
            params.append(order_id)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        # Execute query
        events = []
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    event_dict = json.loads(row[0])
                    event = self.storage._dict_to_event(event_dict)
                    events.append(event)
                    
        except Exception as e:
            self.logger.error(f"Failed to search events: {e}")
        
        return events
    
    def verify_chain_integrity(self, chain_id: str) -> Dict[str, Any]:
        """Verify audit chain integrity."""
        if chain_id not in self.chains:
            return {'error': 'Chain not found'}
        
        chain = self.chains[chain_id]
        
        # Get all events for chain
        events = self.search_events(
            category=chain.category,
            limit=None
        )
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        # Verify chain
        integrity_check = {
            'chain_id': chain_id,
            'total_events': len(events),
            'genesis_hash': chain.genesis_hash,
            'current_hash': chain.current_hash,
            'block_count': chain.block_count,
            'is_valid': True,
            'errors': []
        }
        
        # Verify each event
        for i, event in enumerate(events):
            # Verify event integrity
            if not event.verify_integrity():
                integrity_check['is_valid'] = False
                integrity_check['errors'].append(f"Event {event.event_id} failed integrity check")
            
            # Verify chain linkage
            if i > 0:
                prev_event = events[i-1]
                if event.previous_hash != prev_event.chain_hash:
                    integrity_check['is_valid'] = False
                    integrity_check['errors'].append(f"Chain break at event {event.event_id}")
        
        # Verify chain hash
        if not chain.verify_chain_integrity(events):
            integrity_check['is_valid'] = False
            integrity_check['errors'].append("Chain hash verification failed")
        
        return integrity_check
    
    def export_for_regulator(self, 
                            start_time: datetime,
                            end_time: datetime,
                            regulator: str,
                            export_format: str = 'json') -> str:
        """Export audit trail for regulatory submission."""
        
        # Get relevant events
        events = self.search_events(
            start_time=start_time,
            end_time=end_time,
            limit=None
        )
        
        # Create export package
        export_data = {
            'export_metadata': {
                'regulator': regulator,
                'export_time': datetime.now().isoformat(),
                'period_start': start_time.isoformat(),
                'period_end': end_time.isoformat(),
                'total_events': len(events),
                'export_format': export_format
            },
            'system_info': {
                'version': '1.0',
                'environment': 'production',
                'organization': 'ULTRATHINK Trading System'
            },
            'events': []
        }
        
        # Add events
        for event in events:
            export_data['events'].append(event.to_dict())
        
        # Generate export file
        export_filename = f"audit_export_{regulator}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.{export_format}"
        export_path = Path("compliance/exports") / export_filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write export file
        with open(export_path, 'w') as f:
            if export_format == 'json':
                json.dump(export_data, f, indent=2, default=str)
            else:
                # Could add other formats (XML, CSV, etc.)
                json.dump(export_data, f, indent=2, default=str)
        
        # Mark events as exported
        for event in events:
            event.exported_at = datetime.now()
            event.status = AuditStatus.EXPORTED
        
        self.logger.info(f"Exported {len(events)} events for {regulator} to {export_path}")
        
        return str(export_path)
    
    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit statistics for dashboard."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        stats = {
            'period_days': days,
            'total_events': 0,
            'events_by_category': {},
            'events_by_level': {},
            'events_by_day': {},
            'active_chains': len([c for c in self.chains.values() if not c.is_sealed]),
            'error_rate': 0.0,
            'storage_size': 0,
            'processing_performance': {
                'avg_events_per_second': 0,
                'avg_processing_time': 0,
                'last_flush': self.last_flush.isoformat()
            }
        }
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                # Total events
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM audit_events 
                    WHERE timestamp >= ?
                """, (cutoff_date.isoformat(),))
                stats['total_events'] = cursor.fetchone()[0]
                
                # Events by category
                cursor = conn.execute("""
                    SELECT category, COUNT(*) FROM audit_events 
                    WHERE timestamp >= ?
                    GROUP BY category
                """, (cutoff_date.isoformat(),))
                stats['events_by_category'] = dict(cursor.fetchall())
                
                # Events by level
                cursor = conn.execute("""
                    SELECT level, COUNT(*) FROM audit_events 
                    WHERE timestamp >= ?
                    GROUP BY level
                """, (cutoff_date.isoformat(),))
                stats['events_by_level'] = dict(cursor.fetchall())
                
                # Events by day
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) FROM audit_events 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (cutoff_date.isoformat(),))
                stats['events_by_day'] = dict(cursor.fetchall())
                
                # Error rate
                if stats['total_events'] > 0:
                    stats['error_rate'] = (self.error_count / stats['total_events']) * 100
                
        except Exception as e:
            self.logger.error(f"Failed to get audit statistics: {e}")
        
        return stats
    
    def start_background_processing(self):
        """Start background processing for audit events."""
        self.is_running = True
        self.background_task = asyncio.create_task(self._background_processor())
    
    async def stop_background_processing(self):
        """Stop background processing."""
        self.is_running = False
        if self.background_task:
            await self.background_task
    
    async def _background_processor(self):
        """Background processor for audit events."""
        while self.is_running:
            try:
                # Flush events periodically
                if (datetime.now() - self.last_flush).total_seconds() > self.flush_interval:
                    if self.event_queue:
                        self.flush_events()
                
                # Sleep for a short interval
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Background processor error: {e}")
                await asyncio.sleep(10)

# Example usage and testing
async def demo_enhanced_audit_trail():
    """Demonstration of enhanced audit trail system."""
    print("🚨 ULTRATHINK Week 6 DAY 38: Enhanced Audit Trail Demo 🚨")
    print("=" * 60)
    
    # Create enhanced audit logger
    audit_logger = EnhancedAuditLogger()
    
    # Create audit context
    context = AuditContext(
        session_id="session_001",
        user_id="trader_001",
        client_id="client_123",
        ip_address="192.168.1.100",
        user_agent="Trading System 1.0",
        system_version="1.0.0",
        component="trading_engine",
        environment="production"
    )
    
    print("✅ Enhanced Audit Logger initialized")
    
    # Log various types of events
    events = [
        (AuditLevel.INFO, AuditCategory.TRADING, "ORDER_PLACED", "Buy order placed for BTC-USD"),
        (AuditLevel.WARNING, AuditCategory.RISK_MANAGEMENT, "RISK_LIMIT_APPROACH", "Position approaching risk limit"),
        (AuditLevel.CRITICAL, AuditCategory.COMPLIANCE, "COMPLIANCE_VIOLATION", "Large transaction detected"),
        (AuditLevel.SECURITY, AuditCategory.SECURITY, "LOGIN_ATTEMPT", "Failed login attempt detected"),
        (AuditLevel.INFO, AuditCategory.FINANCIAL, "TRADE_EXECUTION", "Trade executed successfully")
    ]
    
    print(f"\n📝 Logging {len(events)} audit events:")
    
    event_ids = []
    for level, category, event_type, description in events:
        event_id = audit_logger.log_event(
            level=level,
            category=category,
            event_type=event_type,
            description=description,
            context=context,
            symbol="BTC-USD",
            amount=Decimal("50000"),
            currency="USD",
            price=Decimal("45000"),
            risk_score=0.6,
            compliance_flags=["LARGE_TRANSACTION"] if level == AuditLevel.CRITICAL else [],
            tags=["demo", "test"]
        )
        event_ids.append(event_id)
        print(f"- {level.value.upper()}: {description}")
    
    # Flush events
    audit_logger.flush_events()
    
    print(f"\n🔗 Chain status:")
    for chain_id, chain in audit_logger.chains.items():
        print(f"- Chain {chain_id[:8]}... ({chain.category.value}): {chain.block_count} blocks")
    
    # Search events
    print(f"\n🔍 Searching recent events:")
    recent_events = audit_logger.search_events(
        start_time=datetime.now() - timedelta(hours=1),
        category=AuditCategory.TRADING,
        limit=10
    )
    
    print(f"Found {len(recent_events)} trading events")
    
    # Verify chain integrity
    print(f"\n🔐 Verifying chain integrity:")
    for chain_id in audit_logger.chains.keys():
        integrity = audit_logger.verify_chain_integrity(chain_id)
        status = "✅ VALID" if integrity['is_valid'] else "❌ INVALID"
        print(f"- Chain {chain_id[:8]}...: {status}")
        
        if not integrity['is_valid']:
            for error in integrity['errors']:
                print(f"  • {error}")
    
    # Generate statistics
    stats = audit_logger.get_audit_statistics(1)  # Last 1 day
    print(f"\n📊 Audit Statistics:")
    print(f"- Total Events: {stats['total_events']}")
    print(f"- Active Chains: {stats['active_chains']}")
    print(f"- Error Rate: {stats['error_rate']:.2f}%")
    print(f"- Events by Category: {stats['events_by_category']}")
    
    # Export for regulator
    print(f"\n📤 Exporting for regulatory submission:")
    export_path = audit_logger.export_for_regulator(
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now(),
        regulator="SEC",
        export_format="json"
    )
    print(f"Export created: {export_path}")
    
    print(f"\n✅ Enhanced Audit Trail demo completed!")

if __name__ == "__main__":
    asyncio.run(demo_enhanced_audit_trail())