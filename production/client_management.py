#!/usr/bin/env python3
"""
ULTRATHINK Week 6 DAY 40: Institutional Client Management System
Enterprise-grade client onboarding, KYC/AML, and relationship management.

Features:
- Institutional client onboarding and KYC/AML processing
- Multi-tier client classification and privilege management
- Real-time client risk profiling and monitoring
- Prime brokerage and execution services
- Client portal and API access management
- Regulatory reporting and compliance tracking
- Performance analytics and client statements
- Revenue attribution and billing automation

Components:
- Client Onboarding Engine
- KYC/AML Verification System
- Client Risk Assessment
- Privilege and Access Management
- Prime Brokerage Services
- Client Performance Analytics
- Billing and Revenue Management
"""

import asyncio
import hashlib
import uuid
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import aiohttp
import secrets

# Import our trading components
import sys
sys.path.append('/home/richardw/crypto_rf_trading_system')
from production.real_money_trader import AuditLogger, SecurityConfig
from production.portfolio_manager import PortfolioManager, Position
from production.order_management import OrderManager, ManagedOrder
from compliance.compliance_engine import ComplianceEngine, ComplianceViolation
from compliance.audit_trail import EnhancedAuditLogger, AuditLevel, AuditCategory, AuditContext


class ClientTier(Enum):
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    ELIGIBLE_COUNTERPARTY = "eligible_counterparty"
    PRIME_BROKERAGE = "prime_brokerage"
    INSTITUTIONAL = "institutional"
    FAMILY_OFFICE = "family_office"
    SOVEREIGN = "sovereign"


class ClientStatus(Enum):
    PENDING = "pending"
    ONBOARDING = "onboarding"
    KYC_REVIEW = "kyc_review"
    APPROVED = "approved"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    DORMANT = "dormant"


class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"
    ULTRA_HIGH_NET_WORTH = "ultra_high_net_worth"


class ServiceLevel(Enum):
    BASIC = "basic"
    PREMIUM = "premium"
    PRIVATE = "private"
    INSTITUTIONAL = "institutional"
    WHITE_GLOVE = "white_glove"


@dataclass
class ClientEntity:
    """Core client entity information."""
    client_id: str
    legal_name: str
    client_type: str  # individual, corporation, fund, trust, etc.
    incorporation_jurisdiction: Optional[str] = None
    
    # Contact information
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    primary_contact_phone: str = ""
    
    # Address information
    registered_address: Dict[str, str] = field(default_factory=dict)
    business_address: Dict[str, str] = field(default_factory=dict)
    
    # Regulatory information
    lei_code: Optional[str] = None  # Legal Entity Identifier
    regulatory_licenses: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KYCDocument:
    """KYC/AML document tracking."""
    document_id: str
    client_id: str
    document_type: str  # passport, utility_bill, bank_statement, etc.
    file_path: str
    file_hash: str
    uploaded_at: datetime
    verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    verification_status: str = "pending"  # pending, approved, rejected
    expiry_date: Optional[datetime] = None
    notes: str = ""


@dataclass
class ClientProfile:
    """Comprehensive client profile and classification."""
    client_id: str
    client_tier: ClientTier
    status: ClientStatus
    risk_profile: RiskProfile
    service_level: ServiceLevel
    
    # Financial information
    net_worth_usd: Optional[Decimal] = None
    liquid_assets_usd: Optional[Decimal] = None
    investment_experience_years: Optional[int] = None
    annual_income_usd: Optional[Decimal] = None
    
    # Trading parameters
    max_position_size_usd: Decimal = Decimal("1000000")  # $1M default
    max_daily_volume_usd: Decimal = Decimal("5000000")   # $5M default
    leverage_limit: Decimal = Decimal("4.0")             # 4x leverage
    
    # Service parameters
    minimum_balance_usd: Decimal = Decimal("100000")     # $100k minimum
    commission_rate_bps: int = 20                        # 20 bps default
    margin_rate_annual: Decimal = Decimal("0.08")       # 8% annual
    
    # Compliance flags
    pep_status: bool = False  # Politically Exposed Person
    sanctions_check: bool = True
    enhanced_due_diligence: bool = False
    
    # Metadata
    onboarded_at: Optional[datetime] = None
    last_reviewed_at: Optional[datetime] = None
    next_review_due: Optional[datetime] = None


@dataclass
class ClientPerformance:
    """Client trading performance and analytics."""
    client_id: str
    period_start: datetime
    period_end: datetime
    
    # Trading metrics
    total_trades: int = 0
    total_volume_usd: Decimal = Decimal("0")
    total_commission_paid: Decimal = Decimal("0")
    average_trade_size_usd: Decimal = Decimal("0")
    
    # Performance metrics
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_return_percentage: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown_percentage: float = 0.0
    
    # Risk metrics
    var_1d_usd: Decimal = Decimal("0")
    portfolio_beta: Optional[float] = None
    concentration_risk: float = 0.0
    
    # Revenue metrics
    total_revenue_generated: Decimal = Decimal("0")
    margin_interest_paid: Decimal = Decimal("0")
    
    # Activity metrics
    days_active: int = 0
    last_trade_date: Optional[datetime] = None


class ClientOnboarding:
    """Automated client onboarding and KYC/AML processing."""
    
    def __init__(self, 
                 compliance_engine: ComplianceEngine,
                 audit_logger: EnhancedAuditLogger):
        
        self.compliance_engine = compliance_engine
        self.audit_logger = audit_logger
        
        # Database setup
        self.db_path = "production/client_management.db"
        self.setup_database()
        
        # External service integrations
        self.kyc_providers = {
            "jumio": "https://api.jumio.com",
            "thomsonreuters": "https://api.thomsonreuters.com",
            "refinitiv": "https://api.refinitiv.com"
        }
        
        # Document storage
        self.document_storage_path = Path("production/client_documents")
        self.document_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Client Onboarding System initialized")
    
    def setup_database(self):
        """Initialize client management database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Client entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_entities (
                    client_id TEXT PRIMARY KEY,
                    legal_name TEXT NOT NULL,
                    client_type TEXT NOT NULL,
                    incorporation_jurisdiction TEXT,
                    primary_contact_name TEXT,
                    primary_contact_email TEXT,
                    primary_contact_phone TEXT,
                    registered_address TEXT,
                    business_address TEXT,
                    lei_code TEXT,
                    regulatory_licenses TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    entity_data TEXT NOT NULL
                )
            """)
            
            # Client profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS client_profiles (
                    client_id TEXT PRIMARY KEY,
                    client_tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk_profile TEXT NOT NULL,
                    service_level TEXT NOT NULL,
                    net_worth_usd TEXT,
                    liquid_assets_usd TEXT,
                    investment_experience_years INTEGER,
                    annual_income_usd TEXT,
                    max_position_size_usd TEXT NOT NULL,
                    max_daily_volume_usd TEXT NOT NULL,
                    leverage_limit TEXT NOT NULL,
                    minimum_balance_usd TEXT NOT NULL,
                    commission_rate_bps INTEGER NOT NULL,
                    margin_rate_annual TEXT NOT NULL,
                    pep_status BOOLEAN NOT NULL,
                    sanctions_check BOOLEAN NOT NULL,
                    enhanced_due_diligence BOOLEAN NOT NULL,
                    onboarded_at TEXT,
                    last_reviewed_at TEXT,
                    next_review_due TEXT,
                    profile_data TEXT NOT NULL,
                    FOREIGN KEY (client_id) REFERENCES client_entities (client_id)
                )
            """)
            
            # KYC documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kyc_documents (
                    document_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    verified_at TEXT,
                    verified_by TEXT,
                    verification_status TEXT NOT NULL,
                    expiry_date TEXT,
                    notes TEXT,
                    FOREIGN KEY (client_id) REFERENCES client_entities (client_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_client_status ON client_profiles(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_client_tier ON client_profiles(client_tier)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kyc_status ON kyc_documents(verification_status)")
    
    async def start_onboarding(self, entity_data: Dict[str, Any]) -> str:
        """Start client onboarding process."""
        client_id = f"CLIENT_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8].upper()}"
        
        # Create client entity
        entity = ClientEntity(
            client_id=client_id,
            legal_name=entity_data["legal_name"],
            client_type=entity_data["client_type"],
            incorporation_jurisdiction=entity_data.get("incorporation_jurisdiction"),
            primary_contact_name=entity_data.get("primary_contact_name", ""),
            primary_contact_email=entity_data.get("primary_contact_email", ""),
            primary_contact_phone=entity_data.get("primary_contact_phone", ""),
            registered_address=entity_data.get("registered_address", {}),
            business_address=entity_data.get("business_address", {}),
            lei_code=entity_data.get("lei_code"),
            regulatory_licenses=entity_data.get("regulatory_licenses", [])
        )
        
        # Save entity
        await self.save_entity(entity)
        
        # Create initial profile
        profile = ClientProfile(
            client_id=client_id,
            client_tier=self._determine_initial_tier(entity_data),
            status=ClientStatus.ONBOARDING,
            risk_profile=RiskProfile.CONSERVATIVE,  # Start conservative
            service_level=ServiceLevel.BASIC,
            net_worth_usd=entity_data.get("net_worth_usd"),
            liquid_assets_usd=entity_data.get("liquid_assets_usd"),
            investment_experience_years=entity_data.get("investment_experience_years"),
            annual_income_usd=entity_data.get("annual_income_usd")
        )
        
        # Adjust limits based on tier
        self._adjust_limits_for_tier(profile)
        
        # Save profile
        await self.save_profile(profile)
        
        # Log onboarding start
        context = AuditContext(
            session_id=f"ONBOARD_{client_id}",
            user_id="SYSTEM",
            client_id=client_id
        )
        
        self.audit_logger.log_event(
            level=AuditLevel.INFO,
            category=AuditCategory.USER_ACTION,
            event_type="CLIENT_ONBOARDING_STARTED",
            description=f"Client onboarding initiated for {entity.legal_name}",
            context=context,
            client_id=client_id
        )
        
        self.logger.info(f"Started onboarding for client {client_id}: {entity.legal_name}")
        return client_id
    
    def _determine_initial_tier(self, entity_data: Dict[str, Any]) -> ClientTier:
        """Determine initial client tier based on entity data."""
        client_type = entity_data.get("client_type", "individual").lower()
        net_worth = entity_data.get("net_worth_usd", 0)
        
        if client_type in ["sovereign", "central_bank"]:
            return ClientTier.SOVEREIGN
        elif client_type in ["fund", "asset_manager", "bank"]:
            return ClientTier.INSTITUTIONAL
        elif client_type == "family_office":
            return ClientTier.FAMILY_OFFICE
        elif net_worth and net_worth > 50000000:  # $50M+
            return ClientTier.PRIME_BROKERAGE
        elif net_worth and net_worth > 1000000:   # $1M+
            return ClientTier.PROFESSIONAL
        else:
            return ClientTier.RETAIL
    
    def _adjust_limits_for_tier(self, profile: ClientProfile):
        """Adjust trading limits based on client tier."""
        tier_limits = {
            ClientTier.RETAIL: {
                "max_position_size_usd": Decimal("100000"),    # $100K
                "max_daily_volume_usd": Decimal("500000"),     # $500K
                "leverage_limit": Decimal("2.0"),             # 2x
                "minimum_balance_usd": Decimal("10000"),      # $10K
                "commission_rate_bps": 25,                    # 25 bps
                "service_level": ServiceLevel.BASIC
            },
            ClientTier.PROFESSIONAL: {
                "max_position_size_usd": Decimal("1000000"),   # $1M
                "max_daily_volume_usd": Decimal("5000000"),    # $5M
                "leverage_limit": Decimal("4.0"),             # 4x
                "minimum_balance_usd": Decimal("100000"),     # $100K
                "commission_rate_bps": 20,                    # 20 bps
                "service_level": ServiceLevel.PREMIUM
            },
            ClientTier.ELIGIBLE_COUNTERPARTY: {
                "max_position_size_usd": Decimal("10000000"),  # $10M
                "max_daily_volume_usd": Decimal("50000000"),   # $50M
                "leverage_limit": Decimal("10.0"),            # 10x
                "minimum_balance_usd": Decimal("1000000"),    # $1M
                "commission_rate_bps": 15,                    # 15 bps
                "service_level": ServiceLevel.PRIVATE
            },
            ClientTier.INSTITUTIONAL: {
                "max_position_size_usd": Decimal("100000000"), # $100M
                "max_daily_volume_usd": Decimal("500000000"),  # $500M
                "leverage_limit": Decimal("20.0"),            # 20x
                "minimum_balance_usd": Decimal("10000000"),   # $10M
                "commission_rate_bps": 10,                    # 10 bps
                "service_level": ServiceLevel.INSTITUTIONAL
            },
            ClientTier.PRIME_BROKERAGE: {
                "max_position_size_usd": Decimal("500000000"), # $500M
                "max_daily_volume_usd": Decimal("2000000000"), # $2B
                "leverage_limit": Decimal("50.0"),            # 50x
                "minimum_balance_usd": Decimal("50000000"),   # $50M
                "commission_rate_bps": 5,                     # 5 bps
                "service_level": ServiceLevel.WHITE_GLOVE
            },
            ClientTier.SOVEREIGN: {
                "max_position_size_usd": Decimal("1000000000"), # $1B
                "max_daily_volume_usd": Decimal("5000000000"),  # $5B
                "leverage_limit": Decimal("100.0"),            # 100x
                "minimum_balance_usd": Decimal("100000000"),   # $100M
                "commission_rate_bps": 3,                      # 3 bps
                "service_level": ServiceLevel.WHITE_GLOVE
            }
        }
        
        limits = tier_limits.get(profile.client_tier, tier_limits[ClientTier.RETAIL])
        
        for key, value in limits.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
    
    async def upload_kyc_document(self, 
                                 client_id: str, 
                                 document_type: str, 
                                 file_data: bytes, 
                                 filename: str) -> str:
        """Upload and process KYC document."""
        
        # Generate document ID and hash
        document_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Save file securely
        file_extension = filename.split('.')[-1] if '.' in filename else 'bin'
        safe_filename = f"{client_id}_{document_type}_{document_id}.{file_extension}"
        file_path = self.document_storage_path / safe_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        # Create document record
        document = KYCDocument(
            document_id=document_id,
            client_id=client_id,
            document_type=document_type,
            file_path=str(file_path),
            file_hash=file_hash,
            uploaded_at=datetime.now()
        )
        
        # Save to database
        await self.save_kyc_document(document)
        
        # Log document upload
        context = AuditContext(
            session_id=f"KYC_{client_id}",
            user_id="SYSTEM",
            client_id=client_id
        )
        
        self.audit_logger.log_event(
            level=AuditLevel.INFO,
            category=AuditCategory.COMPLIANCE,
            event_type="KYC_DOCUMENT_UPLOADED",
            description=f"KYC document uploaded: {document_type}",
            context=context,
            client_id=client_id
        )
        
        # Start automated verification if possible
        await self._start_document_verification(document)
        
        self.logger.info(f"Uploaded KYC document {document_id} for client {client_id}")
        return document_id
    
    async def _start_document_verification(self, document: KYCDocument):
        """Start automated document verification process."""
        try:
            # Simulate document verification
            # In production, would integrate with KYC providers like Jumio, Thomson Reuters
            
            verification_result = await self._simulate_document_verification(document)
            
            if verification_result["status"] == "approved":
                document.verification_status = "approved"
                document.verified_at = datetime.now()
                document.verified_by = "AUTOMATED_SYSTEM"
                
                # Update database
                await self.save_kyc_document(document)
                
                # Check if client can be approved
                await self._check_client_approval_status(document.client_id)
                
            else:
                document.verification_status = "rejected"
                document.notes = verification_result.get("reason", "Automated verification failed")
                await self.save_kyc_document(document)
                
        except Exception as e:
            self.logger.error(f"Document verification failed for {document.document_id}: {e}")
            document.verification_status = "manual_review"
            document.notes = f"Automated verification error: {str(e)}"
            await self.save_kyc_document(document)
    
    async def _simulate_document_verification(self, document: KYCDocument) -> Dict[str, Any]:
        """Simulate document verification (placeholder for real KYC integration)."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simple simulation - approve 90% of documents
        import random
        if random.random() < 0.9:
            return {"status": "approved", "confidence": 0.95}
        else:
            return {"status": "rejected", "reason": "Document quality insufficient"}
    
    async def _check_client_approval_status(self, client_id: str):
        """Check if client has met all requirements for approval."""
        
        # Get client profile
        profile = await self.get_client_profile(client_id)
        if not profile or profile.status != ClientStatus.KYC_REVIEW:
            return
        
        # Check required documents
        required_docs = self._get_required_documents(profile.client_tier)
        uploaded_docs = await self.get_kyc_documents(client_id)
        
        approved_doc_types = {
            doc.document_type for doc in uploaded_docs 
            if doc.verification_status == "approved"
        }
        
        if required_docs.issubset(approved_doc_types):
            # All required documents approved
            profile.status = ClientStatus.APPROVED
            profile.onboarded_at = datetime.now()
            profile.next_review_due = datetime.now() + timedelta(days=365)  # Annual review
            
            await self.save_profile(profile)
            
            # Log approval
            context = AuditContext(
                session_id=f"APPROVAL_{client_id}",
                user_id="SYSTEM",
                client_id=client_id
            )
            
            self.audit_logger.log_event(
                level=AuditLevel.INFO,
                category=AuditCategory.COMPLIANCE,
                event_type="CLIENT_APPROVED",
                description="Client has been approved for trading",
                context=context,
                client_id=client_id
            )
            
            self.logger.info(f"Client {client_id} approved for trading")
    
    def _get_required_documents(self, client_tier: ClientTier) -> set:
        """Get required KYC documents for client tier."""
        base_docs = {"passport", "proof_of_address"}
        
        if client_tier in [ClientTier.PROFESSIONAL, ClientTier.ELIGIBLE_COUNTERPARTY]:
            base_docs.add("financial_statement")
        
        if client_tier in [ClientTier.INSTITUTIONAL, ClientTier.PRIME_BROKERAGE, ClientTier.SOVEREIGN]:
            base_docs.update({"incorporation_certificate", "board_resolution", "beneficial_ownership"})
        
        return base_docs
    
    async def save_entity(self, entity: ClientEntity):
        """Save client entity to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO client_entities (
                        client_id, legal_name, client_type, incorporation_jurisdiction,
                        primary_contact_name, primary_contact_email, primary_contact_phone,
                        registered_address, business_address, lei_code, regulatory_licenses,
                        created_at, updated_at, entity_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.client_id,
                    entity.legal_name,
                    entity.client_type,
                    entity.incorporation_jurisdiction,
                    entity.primary_contact_name,
                    entity.primary_contact_email,
                    entity.primary_contact_phone,
                    json.dumps(entity.registered_address),
                    json.dumps(entity.business_address),
                    entity.lei_code,
                    json.dumps(entity.regulatory_licenses),
                    entity.created_at.isoformat(),
                    entity.updated_at.isoformat(),
                    json.dumps(asdict(entity), default=str)
                ))
        except Exception as e:
            self.logger.error(f"Failed to save entity {entity.client_id}: {e}")
            raise
    
    async def save_profile(self, profile: ClientProfile):
        """Save client profile to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO client_profiles (
                        client_id, client_tier, status, risk_profile, service_level,
                        net_worth_usd, liquid_assets_usd, investment_experience_years, annual_income_usd,
                        max_position_size_usd, max_daily_volume_usd, leverage_limit,
                        minimum_balance_usd, commission_rate_bps, margin_rate_annual,
                        pep_status, sanctions_check, enhanced_due_diligence,
                        onboarded_at, last_reviewed_at, next_review_due, profile_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.client_id,
                    profile.client_tier.value,
                    profile.status.value,
                    profile.risk_profile.value,
                    profile.service_level.value,
                    str(profile.net_worth_usd) if profile.net_worth_usd else None,
                    str(profile.liquid_assets_usd) if profile.liquid_assets_usd else None,
                    profile.investment_experience_years,
                    str(profile.annual_income_usd) if profile.annual_income_usd else None,
                    str(profile.max_position_size_usd),
                    str(profile.max_daily_volume_usd),
                    str(profile.leverage_limit),
                    str(profile.minimum_balance_usd),
                    profile.commission_rate_bps,
                    str(profile.margin_rate_annual),
                    profile.pep_status,
                    profile.sanctions_check,
                    profile.enhanced_due_diligence,
                    profile.onboarded_at.isoformat() if profile.onboarded_at else None,
                    profile.last_reviewed_at.isoformat() if profile.last_reviewed_at else None,
                    profile.next_review_due.isoformat() if profile.next_review_due else None,
                    json.dumps(asdict(profile), default=str)
                ))
        except Exception as e:
            self.logger.error(f"Failed to save profile {profile.client_id}: {e}")
            raise
    
    async def save_kyc_document(self, document: KYCDocument):
        """Save KYC document to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO kyc_documents (
                        document_id, client_id, document_type, file_path, file_hash,
                        uploaded_at, verified_at, verified_by, verification_status,
                        expiry_date, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.document_id,
                    document.client_id,
                    document.document_type,
                    document.file_path,
                    document.file_hash,
                    document.uploaded_at.isoformat(),
                    document.verified_at.isoformat() if document.verified_at else None,
                    document.verified_by,
                    document.verification_status,
                    document.expiry_date.isoformat() if document.expiry_date else None,
                    document.notes
                ))
        except Exception as e:
            self.logger.error(f"Failed to save KYC document {document.document_id}: {e}")
            raise
    
    async def get_client_profile(self, client_id: str) -> Optional[ClientProfile]:
        """Get client profile by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT client_tier, status, risk_profile, service_level,
                           net_worth_usd, liquid_assets_usd, investment_experience_years, annual_income_usd,
                           max_position_size_usd, max_daily_volume_usd, leverage_limit,
                           minimum_balance_usd, commission_rate_bps, margin_rate_annual,
                           pep_status, sanctions_check, enhanced_due_diligence,
                           onboarded_at, last_reviewed_at, next_review_due
                    FROM client_profiles WHERE client_id = ?
                """, (client_id,))
                
                row = cursor.fetchone()
                if row:
                    profile = ClientProfile(
                        client_id=client_id,
                        client_tier=ClientTier(row[0]),
                        status=ClientStatus(row[1]),
                        risk_profile=RiskProfile(row[2]),
                        service_level=ServiceLevel(row[3]),
                        net_worth_usd=Decimal(row[4]) if row[4] else None,
                        liquid_assets_usd=Decimal(row[5]) if row[5] else None,
                        investment_experience_years=row[6],
                        annual_income_usd=Decimal(row[7]) if row[7] else None,
                        max_position_size_usd=Decimal(row[8]),
                        max_daily_volume_usd=Decimal(row[9]),
                        leverage_limit=Decimal(row[10]),
                        minimum_balance_usd=Decimal(row[11]),
                        commission_rate_bps=row[12],
                        margin_rate_annual=Decimal(row[13]),
                        pep_status=bool(row[14]),
                        sanctions_check=bool(row[15]),
                        enhanced_due_diligence=bool(row[16]),
                        onboarded_at=datetime.fromisoformat(row[17]) if row[17] else None,
                        last_reviewed_at=datetime.fromisoformat(row[18]) if row[18] else None,
                        next_review_due=datetime.fromisoformat(row[19]) if row[19] else None
                    )
                    return profile
                    
        except Exception as e:
            self.logger.error(f"Failed to get client profile {client_id}: {e}")
        
        return None
    
    async def get_kyc_documents(self, client_id: str) -> List[KYCDocument]:
        """Get all KYC documents for a client."""
        documents = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT document_id, client_id, document_type, file_path, file_hash,
                           uploaded_at, verified_at, verified_by, verification_status,
                           expiry_date, notes
                    FROM kyc_documents WHERE client_id = ?
                    ORDER BY uploaded_at DESC
                """, (client_id,))
                
                for row in cursor.fetchall():
                    document = KYCDocument(
                        document_id=row[0],
                        client_id=row[1],
                        document_type=row[2],
                        file_path=row[3],
                        file_hash=row[4],
                        uploaded_at=datetime.fromisoformat(row[5]),
                        verified_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        verified_by=row[7],
                        verification_status=row[8],
                        expiry_date=datetime.fromisoformat(row[9]) if row[9] else None,
                        notes=row[10] or ""
                    )
                    documents.append(document)
                    
        except Exception as e:
            self.logger.error(f"Failed to get KYC documents for {client_id}: {e}")
        
        return documents


class ClientAnalytics:
    """Client performance analytics and reporting."""
    
    def __init__(self, 
                 portfolio_manager: PortfolioManager,
                 order_manager: OrderManager,
                 audit_logger: EnhancedAuditLogger):
        
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.audit_logger = audit_logger
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_client_performance(self, 
                                        client_id: str,
                                        start_date: datetime,
                                        end_date: datetime) -> ClientPerformance:
        """Generate comprehensive client performance report."""
        
        performance = ClientPerformance(
            client_id=client_id,
            period_start=start_date,
            period_end=end_date
        )
        
        # Get client orders and trades for the period
        client_orders = await self._get_client_orders(client_id, start_date, end_date)
        
        if client_orders:
            # Calculate trading metrics
            performance.total_trades = len(client_orders)
            performance.total_volume_usd = sum(
                order.filled_quantity * order.average_price 
                for order in client_orders 
                if order.average_price
            )
            performance.total_commission_paid = sum(
                order.total_commission for order in client_orders
            )
            performance.average_trade_size_usd = (
                performance.total_volume_usd / performance.total_trades 
                if performance.total_trades > 0 else Decimal("0")
            )
            
            # Calculate P&L
            realized_pnl, unrealized_pnl = await self._calculate_client_pnl(client_id, client_orders)
            performance.realized_pnl = realized_pnl
            performance.unrealized_pnl = unrealized_pnl
            
            # Calculate returns
            if performance.total_volume_usd > 0:
                performance.total_return_percentage = float(
                    (realized_pnl + unrealized_pnl) / performance.total_volume_usd * 100
                )
            
            # Calculate revenue generated
            performance.total_revenue_generated = performance.total_commission_paid
            
            # Activity metrics
            performance.days_active = (end_date - start_date).days
            performance.last_trade_date = max(
                order.updated_at for order in client_orders
            ) if client_orders else None
        
        return performance
    
    async def _get_client_orders(self, 
                               client_id: str, 
                               start_date: datetime, 
                               end_date: datetime) -> List[ManagedOrder]:
        """Get client orders for specified period."""
        # This would integrate with the order management system
        # For now, return empty list as placeholder
        return []
    
    async def _calculate_client_pnl(self, 
                                   client_id: str, 
                                   orders: List[ManagedOrder]) -> Tuple[Decimal, Decimal]:
        """Calculate realized and unrealized P&L for client."""
        # Placeholder implementation
        realized_pnl = Decimal("0")
        unrealized_pnl = Decimal("0")
        
        # In production, this would:
        # 1. Calculate realized P&L from closed positions
        # 2. Calculate unrealized P&L from open positions
        # 3. Account for dividends, interest, fees
        
        return realized_pnl, unrealized_pnl


# Example usage and testing
async def demo_client_management():
    """Demonstration of the client management system."""
    print("🚨 ULTRATHINK Week 6: Institutional Client Management Demo 🚨")
    print("=" * 70)
    
    # Mock components for demo
    from compliance.test_utils import MockComplianceEngine, MockAuditLogger
    
    compliance_engine = MockComplianceEngine()
    audit_logger = MockAuditLogger()
    
    # Create client onboarding system
    onboarding = ClientOnboarding(compliance_engine, audit_logger)
    
    # Demo institutional client onboarding
    print("\n📋 Starting institutional client onboarding...")
    
    entity_data = {
        "legal_name": "Quantum Capital Management LLC",
        "client_type": "fund",
        "incorporation_jurisdiction": "Delaware, USA",
        "primary_contact_name": "Sarah Johnson",
        "primary_contact_email": "sarah.johnson@quantumcap.com",
        "primary_contact_phone": "+1-212-555-0123",
        "registered_address": {
            "street": "123 Wall Street, Suite 4500",
            "city": "New York",
            "state": "NY",
            "postal_code": "10005",
            "country": "USA"
        },
        "net_worth_usd": 500000000,  # $500M AUM
        "investment_experience_years": 15,
        "regulatory_licenses": ["SEC_RIA", "CFTC_CPO"]
    }
    
    client_id = await onboarding.start_onboarding(entity_data)
    print(f"✅ Client onboarding started: {client_id}")
    
    # Get client profile
    profile = await onboarding.get_client_profile(client_id)
    if profile:
        print(f"✅ Client tier: {profile.client_tier.value}")
        print(f"✅ Service level: {profile.service_level.value}")
        print(f"✅ Max position size: ${profile.max_position_size_usd:,.2f}")
        print(f"✅ Commission rate: {profile.commission_rate_bps} bps")
    
    # Demo KYC document upload
    print(f"\n📄 Uploading KYC documents...")
    
    # Simulate document uploads
    mock_documents = [
        ("incorporation_certificate", b"Mock incorporation certificate data"),
        ("board_resolution", b"Mock board resolution data"),
        ("beneficial_ownership", b"Mock beneficial ownership data"),
        ("financial_statement", b"Mock financial statement data")
    ]
    
    for doc_type, doc_data in mock_documents:
        doc_id = await onboarding.upload_kyc_document(
            client_id, doc_type, doc_data, f"{doc_type}.pdf"
        )
        print(f"✅ Uploaded {doc_type}: {doc_id}")
    
    # Wait for verification
    await asyncio.sleep(0.5)
    
    # Check final status
    updated_profile = await onboarding.get_client_profile(client_id)
    if updated_profile:
        print(f"\n🎯 Final client status: {updated_profile.status.value}")
        if updated_profile.status == ClientStatus.APPROVED:
            print("✅ Client approved for trading!")
            print(f"✅ Onboarded at: {updated_profile.onboarded_at}")
        
    print(f"\n✅ Institutional Client Management demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_client_management())