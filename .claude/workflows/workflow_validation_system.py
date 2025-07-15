#!/usr/bin/env python3
"""
ULTRATHINK Workflow Validation System
Comprehensive testing and validation framework for workflow templates

Philosophy: Validate workflow templates for correctness, performance, and trading relevance
Performance: < 5 seconds per template validation
Intelligence: Multi-dimensional validation with trading-specific checks
"""

import json
import time
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Import workflow template engine
from workflow_template_engine import (
    WorkflowTemplateEngine, WorkflowTemplate, 
    WorkflowStep, WorkflowType, WorkflowComplexity
)

class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"

class ValidationResult(Enum):
    """Validation result types"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    issue_id: str
    severity: ValidationResult
    category: str
    message: str
    step_id: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None

@dataclass
class ValidationReport:
    """Complete validation report for a workflow template"""
    template_id: str
    template_name: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    validation_score: float  # 0.0 - 1.0
    issues: List[ValidationIssue]
    performance_metrics: Dict[str, float]
    trading_relevance_score: float
    code_quality_score: float
    validation_time_ms: float
    timestamp: datetime
    recommendations: List[str]

class CodeValidator:
    """Validates Python code in workflow templates"""
    
    def __init__(self):
        self.TRADING_KEYWORDS = [
            'kelly', 'position', 'risk', 'portfolio', 'sharpe', 'return',
            'drawdown', 'volatility', 'signal', 'trade', 'buy', 'sell',
            'momentum', 'rsi', 'macd', 'stop_loss', 'take_profit'
        ]
        
        self.DANGEROUS_PATTERNS = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'open\s*\([^)]*["\']w["\']',
            r'os\.system\s*\(',
            r'subprocess\.',
            r'rm\s+-rf',
            r'delete\s+from',
            r'drop\s+table'
        ]
        
        self.PERFORMANCE_PATTERNS = [
            r'for\s+\w+\s+in\s+range\s*\(\s*\d{4,}',  # Large loops
            r'while\s+True\s*:',  # Infinite loops
            r'time\.sleep\s*\(\s*\d{2,}',  # Long sleeps
            r'\.iterrows\s*\(\s*\)',  # Inefficient pandas iteration
        ]
    
    def validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax"""
        issues = []
        
        try:
            ast.parse(code)
            return True, issues
        except SyntaxError as e:
            issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, issues
        except Exception as e:
            issues.append(f"Code parsing error: {str(e)}")
            return False, issues
    
    def check_dangerous_patterns(self, code: str) -> List[str]:
        """Check for dangerous code patterns"""
        issues = []
        
        for pattern in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        return issues
    
    def check_performance_issues(self, code: str) -> List[str]:
        """Check for potential performance issues"""
        issues = []
        
        for pattern in self.PERFORMANCE_PATTERNS:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                issues.append(f"Performance concern: {pattern}")
        
        return issues
    
    def calculate_trading_relevance(self, code: str) -> float:
        """Calculate trading relevance score"""
        code_lower = code.lower()
        keyword_count = sum(1 for keyword in self.TRADING_KEYWORDS if keyword in code_lower)
        
        # Normalize by code length
        code_length = len(code.split())
        if code_length == 0:
            return 0.0
        
        relevance_score = min(1.0, keyword_count / (code_length * 0.1))
        return relevance_score
    
    def validate_code_quality(self, code: str) -> Tuple[float, List[str]]:
        """Validate code quality metrics"""
        issues = []
        quality_score = 1.0
        
        # Check for docstrings
        if '"""' not in code and "'''" not in code:
            issues.append("Missing docstring")
            quality_score -= 0.1
        
        # Check for proper error handling
        if 'try:' not in code and 'except:' not in code:
            issues.append("No error handling detected")
            quality_score -= 0.1
        
        # Check for hardcoded values
        hardcoded_numbers = re.findall(r'\b\d{4,}\b', code)
        if hardcoded_numbers:
            issues.append(f"Hardcoded values detected: {hardcoded_numbers}")
            quality_score -= 0.05
        
        # Check for proper imports
        if 'import' not in code:
            issues.append("No imports detected - may be incomplete")
            quality_score -= 0.1
        
        return max(0.0, quality_score), issues

class WorkflowStepValidator:
    """Validates individual workflow steps"""
    
    def __init__(self):
        self.code_validator = CodeValidator()
        
        self.REQUIRED_FIELDS = [
            'step_id', 'step_name', 'step_type', 'description',
            'code_template', 'required_context', 'dependencies',
            'validation_checks', 'estimated_time_minutes'
        ]
        
        self.VALID_STEP_TYPES = [
            'code', 'config', 'validation', 'execution', 'analysis'
        ]
        
        self.TRADING_STAGES = [
            'SETUP', 'DATA_INGESTION', 'FEATURE_ENGINEERING', 'MODEL_TRAINING',
            'SIGNAL_GENERATION', 'RISK_ASSESSMENT', 'ORDER_EXECUTION',
            'PERFORMANCE_ANALYSIS', 'MONITORING'
        ]
    
    def validate_step(self, step: WorkflowStep) -> List[ValidationIssue]:
        """Validate a single workflow step"""
        issues = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if not hasattr(step, field) or getattr(step, field) is None:
                issues.append(ValidationIssue(
                    issue_id=f"missing_{field}",
                    severity=ValidationResult.FAIL,
                    category="completeness",
                    message=f"Missing required field: {field}",
                    step_id=step.step_id,
                    suggestion=f"Add {field} to step definition"
                ))
        
        # Validate step type
        if step.step_type not in self.VALID_STEP_TYPES:
            issues.append(ValidationIssue(
                issue_id="invalid_step_type",
                severity=ValidationResult.FAIL,
                category="configuration",
                message=f"Invalid step type: {step.step_type}",
                step_id=step.step_id,
                suggestion=f"Use one of: {', '.join(self.VALID_STEP_TYPES)}"
            ))
        
        # Validate trading stage
        if hasattr(step, 'trading_stage') and step.trading_stage not in self.TRADING_STAGES:
            issues.append(ValidationIssue(
                issue_id="invalid_trading_stage",
                severity=ValidationResult.WARNING,
                category="configuration",
                message=f"Invalid trading stage: {step.trading_stage}",
                step_id=step.step_id,
                suggestion=f"Use one of: {', '.join(self.TRADING_STAGES)}"
            ))
        
        # Validate code template
        if step.code_template:
            syntax_valid, syntax_issues = self.code_validator.validate_python_syntax(step.code_template)
            if not syntax_valid:
                for issue in syntax_issues:
                    issues.append(ValidationIssue(
                        issue_id="syntax_error",
                        severity=ValidationResult.FAIL,
                        category="code_quality",
                        message=issue,
                        step_id=step.step_id,
                        suggestion="Fix Python syntax errors"
                    ))
            
            # Check for dangerous patterns
            dangerous_patterns = self.code_validator.check_dangerous_patterns(step.code_template)
            for pattern in dangerous_patterns:
                issues.append(ValidationIssue(
                    issue_id="dangerous_pattern",
                    severity=ValidationResult.FAIL,
                    category="security",
                    message=pattern,
                    step_id=step.step_id,
                    suggestion="Remove dangerous code patterns"
                ))
            
            # Check performance issues
            performance_issues = self.code_validator.check_performance_issues(step.code_template)
            for issue in performance_issues:
                issues.append(ValidationIssue(
                    issue_id="performance_issue",
                    severity=ValidationResult.WARNING,
                    category="performance",
                    message=issue,
                    step_id=step.step_id,
                    suggestion="Optimize code for better performance"
                ))
        
        # Validate time estimates
        if step.estimated_time_minutes <= 0:
            issues.append(ValidationIssue(
                issue_id="invalid_time_estimate",
                severity=ValidationResult.WARNING,
                category="configuration",
                message="Time estimate must be positive",
                step_id=step.step_id,
                suggestion="Set realistic time estimate in minutes"
            ))
        
        # Check for extremely long steps
        if step.estimated_time_minutes > 1440:  # 24 hours
            issues.append(ValidationIssue(
                issue_id="excessive_time_estimate",
                severity=ValidationResult.WARNING,
                category="configuration",
                message="Step estimate exceeds 24 hours",
                step_id=step.step_id,
                suggestion="Consider breaking into smaller steps"
            ))
        
        return issues

class WorkflowTemplateValidator:
    """Validates complete workflow templates"""
    
    def __init__(self):
        self.step_validator = WorkflowStepValidator()
        self.code_validator = CodeValidator()
    
    def validate_template(self, template: WorkflowTemplate, 
                         validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
        """Validate complete workflow template"""
        
        print(f"🔍 Validating template: {template.workflow_name}")
        start_time = time.time()
        
        issues = []
        
        # Basic template validation
        issues.extend(self._validate_template_structure(template))
        
        # Step validation
        for step in template.steps:
            issues.extend(self.step_validator.validate_step(step))
        
        # Template-level validation
        issues.extend(self._validate_template_logic(template))
        
        if validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            issues.extend(self._validate_template_dependencies(template))
            issues.extend(self._validate_template_performance(template))
        
        # Calculate scores
        validation_score = self._calculate_validation_score(issues)
        trading_relevance_score = self._calculate_trading_relevance(template)
        code_quality_score = self._calculate_code_quality(template)
        
        # Determine overall result
        overall_result = self._determine_overall_result(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, template)
        
        # Performance metrics
        validation_time_ms = (time.time() - start_time) * 1000
        performance_metrics = {
            'validation_time_ms': validation_time_ms,
            'issues_count': len(issues),
            'steps_validated': len(template.steps)
        }
        
        validation_report = ValidationReport(
            template_id=template.workflow_id,
            template_name=template.workflow_name,
            validation_level=validation_level,
            overall_result=overall_result,
            validation_score=validation_score,
            issues=issues,
            performance_metrics=performance_metrics,
            trading_relevance_score=trading_relevance_score,
            code_quality_score=code_quality_score,
            validation_time_ms=validation_time_ms,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
        
        print(f"✅ Validation complete in {validation_time_ms:.1f}ms")
        print(f"📊 Score: {validation_score:.2f}, Issues: {len(issues)}")
        
        return validation_report
    
    def _validate_template_structure(self, template: WorkflowTemplate) -> List[ValidationIssue]:
        """Validate basic template structure"""
        issues = []
        
        # Check required fields
        if not template.workflow_name:
            issues.append(ValidationIssue(
                issue_id="missing_workflow_name",
                severity=ValidationResult.FAIL,
                category="structure",
                message="Workflow name is required",
                suggestion="Add descriptive workflow name"
            ))
        
        if not template.steps:
            issues.append(ValidationIssue(
                issue_id="no_steps",
                severity=ValidationResult.FAIL,
                category="structure",
                message="Workflow has no steps",
                suggestion="Add at least one workflow step"
            ))
        
        # Check step count
        if len(template.steps) > 20:
            issues.append(ValidationIssue(
                issue_id="too_many_steps",
                severity=ValidationResult.WARNING,
                category="structure",
                message="Workflow has too many steps (>20)",
                suggestion="Consider breaking into sub-workflows"
            ))
        
        return issues
    
    def _validate_template_logic(self, template: WorkflowTemplate) -> List[ValidationIssue]:
        """Validate template logic and flow"""
        issues = []
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in template.steps]
        if len(step_ids) != len(set(step_ids)):
            issues.append(ValidationIssue(
                issue_id="duplicate_step_ids",
                severity=ValidationResult.FAIL,
                category="logic",
                message="Duplicate step IDs found",
                suggestion="Ensure all step IDs are unique"
            ))
        
        # Check dependency chain
        all_step_ids = set(step_ids)
        for step in template.steps:
            for dep in step.dependencies:
                if dep in all_step_ids:
                    # Check for circular dependencies
                    if self._has_circular_dependency(template.steps, step.step_id, dep):
                        issues.append(ValidationIssue(
                            issue_id="circular_dependency",
                            severity=ValidationResult.FAIL,
                            category="logic",
                            message=f"Circular dependency detected: {step.step_id} <-> {dep}",
                            step_id=step.step_id,
                            suggestion="Remove circular dependencies"
                        ))
        
        return issues
    
    def _validate_template_dependencies(self, template: WorkflowTemplate) -> List[ValidationIssue]:
        """Validate template dependencies"""
        issues = []
        
        # Check for common Python packages
        all_dependencies = set()
        for step in template.steps:
            all_dependencies.update(step.dependencies)
        
        # Check for missing standard dependencies
        has_pandas = any('pandas' in dep for dep in all_dependencies)
        has_numpy = any('numpy' in dep for dep in all_dependencies)
        
        if not has_pandas and template.workflow_type in [
            WorkflowType.STRATEGY_BACKTESTING, WorkflowType.ENSEMBLE_TRAINING
        ]:
            issues.append(ValidationIssue(
                issue_id="missing_pandas",
                severity=ValidationResult.WARNING,
                category="dependencies",
                message="pandas not found in dependencies but likely needed",
                suggestion="Add pandas to step dependencies"
            ))
        
        if not has_numpy and template.workflow_type in [
            WorkflowType.KELLY_OPTIMIZATION, WorkflowType.RISK_MANAGEMENT
        ]:
            issues.append(ValidationIssue(
                issue_id="missing_numpy",
                severity=ValidationResult.WARNING,
                category="dependencies",
                message="numpy not found in dependencies but likely needed",
                suggestion="Add numpy to step dependencies"
            ))
        
        return issues
    
    def _validate_template_performance(self, template: WorkflowTemplate) -> List[ValidationIssue]:
        """Validate template performance characteristics"""
        issues = []
        
        # Check total estimated time
        total_time = sum(step.estimated_time_minutes for step in template.steps)
        if total_time > 1440:  # 24 hours
            issues.append(ValidationIssue(
                issue_id="excessive_total_time",
                severity=ValidationResult.WARNING,
                category="performance",
                message=f"Total estimated time exceeds 24 hours: {total_time} minutes",
                suggestion="Consider parallel execution or optimization"
            ))
        
        # Check for performance targets
        if not template.performance_targets:
            issues.append(ValidationIssue(
                issue_id="no_performance_targets",
                severity=ValidationResult.WARNING,
                category="performance",
                message="No performance targets defined",
                suggestion="Add performance targets for measurement"
            ))
        
        return issues
    
    def _has_circular_dependency(self, steps: List[WorkflowStep], step_id: str, dependency: str) -> bool:
        """Check for circular dependencies between steps"""
        # Simple circular dependency check
        for step in steps:
            if step.step_id == dependency:
                return step_id in step.dependencies
        return False
    
    def _calculate_validation_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall validation score"""
        if not issues:
            return 1.0
        
        # Weight different severity levels
        severity_weights = {
            ValidationResult.FAIL: -0.3,
            ValidationResult.ERROR: -0.2,
            ValidationResult.WARNING: -0.1,
            ValidationResult.PASS: 0.0
        }
        
        score = 1.0
        for issue in issues:
            score += severity_weights.get(issue.severity, -0.1)
        
        return max(0.0, score)
    
    def _calculate_trading_relevance(self, template: WorkflowTemplate) -> float:
        """Calculate trading relevance score"""
        total_relevance = 0.0
        
        for step in template.steps:
            if step.code_template:
                relevance = self.code_validator.calculate_trading_relevance(step.code_template)
                total_relevance += relevance
        
        return total_relevance / len(template.steps) if template.steps else 0.0
    
    def _calculate_code_quality(self, template: WorkflowTemplate) -> float:
        """Calculate code quality score"""
        total_quality = 0.0
        
        for step in template.steps:
            if step.code_template:
                quality, _ = self.code_validator.validate_code_quality(step.code_template)
                total_quality += quality
        
        return total_quality / len(template.steps) if template.steps else 0.0
    
    def _determine_overall_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result"""
        if not issues:
            return ValidationResult.PASS
        
        # Check for any FAIL issues
        if any(issue.severity == ValidationResult.FAIL for issue in issues):
            return ValidationResult.FAIL
        
        # Check for any ERROR issues
        if any(issue.severity == ValidationResult.ERROR for issue in issues):
            return ValidationResult.ERROR
        
        # Check for WARNING issues
        if any(issue.severity == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING
        
        return ValidationResult.PASS
    
    def _generate_recommendations(self, issues: List[ValidationIssue], 
                                template: WorkflowTemplate) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Group issues by category
        issue_categories = {}
        for issue in issues:
            if issue.category not in issue_categories:
                issue_categories[issue.category] = []
            issue_categories[issue.category].append(issue)
        
        # Generate category-specific recommendations
        if 'code_quality' in issue_categories:
            recommendations.append("Improve code quality with proper docstrings and error handling")
        
        if 'performance' in issue_categories:
            recommendations.append("Optimize performance by reducing complexity and adding parallel execution")
        
        if 'security' in issue_categories:
            recommendations.append("Remove dangerous code patterns and add security validations")
        
        if 'dependencies' in issue_categories:
            recommendations.append("Review and complete dependency specifications")
        
        # Template-specific recommendations
        if template.workflow_type == WorkflowType.LIVE_DEPLOYMENT:
            recommendations.append("Add comprehensive monitoring and alerting for live deployment")
        
        if template.workflow_type == WorkflowType.RISK_MANAGEMENT:
            recommendations.append("Implement additional risk validation checks and limits")
        
        return recommendations

class WorkflowValidationSuite:
    """Complete validation suite for workflow templates"""
    
    def __init__(self):
        self.template_validator = WorkflowTemplateValidator()
        self.validation_reports = []
    
    def validate_all_templates(self, templates: List[WorkflowTemplate], 
                             validation_level: ValidationLevel = ValidationLevel.STANDARD) -> List[ValidationReport]:
        """Validate all workflow templates"""
        print(f"🔍 Starting validation suite for {len(templates)} templates")
        print(f"📊 Validation level: {validation_level.value}")
        print("=" * 60)
        
        reports = []
        
        for i, template in enumerate(templates, 1):
            print(f"\n🔍 Validating template {i}/{len(templates)}: {template.workflow_name}")
            
            try:
                report = self.template_validator.validate_template(template, validation_level)
                reports.append(report)
                
                # Print summary
                result_emoji = {
                    ValidationResult.PASS: "✅",
                    ValidationResult.WARNING: "⚠️",
                    ValidationResult.ERROR: "❌",
                    ValidationResult.FAIL: "❌"
                }
                
                print(f"{result_emoji[report.overall_result]} {report.overall_result.value.upper()}")
                print(f"📊 Score: {report.validation_score:.2f}")
                print(f"🎯 Trading relevance: {report.trading_relevance_score:.2f}")
                print(f"💎 Code quality: {report.code_quality_score:.2f}")
                
                if report.issues:
                    print(f"⚠️  Issues found: {len(report.issues)}")
                    for issue in report.issues[:3]:  # Show first 3 issues
                        print(f"  - {issue.severity.value}: {issue.message}")
                    if len(report.issues) > 3:
                        print(f"  ... and {len(report.issues) - 3} more")
                
            except Exception as e:
                print(f"❌ Validation failed: {e}")
                # Create error report
                error_report = ValidationReport(
                    template_id=template.workflow_id,
                    template_name=template.workflow_name,
                    validation_level=validation_level,
                    overall_result=ValidationResult.ERROR,
                    validation_score=0.0,
                    issues=[ValidationIssue(
                        issue_id="validation_error",
                        severity=ValidationResult.ERROR,
                        category="system",
                        message=f"Validation system error: {str(e)}"
                    )],
                    performance_metrics={},
                    trading_relevance_score=0.0,
                    code_quality_score=0.0,
                    validation_time_ms=0.0,
                    timestamp=datetime.now(),
                    recommendations=["Fix validation system error"]
                )
                reports.append(error_report)
        
        self.validation_reports = reports
        self._print_validation_summary(reports)
        
        return reports
    
    def _print_validation_summary(self, reports: List[ValidationReport]):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total_templates = len(reports)
        passed = sum(1 for r in reports if r.overall_result == ValidationResult.PASS)
        warnings = sum(1 for r in reports if r.overall_result == ValidationResult.WARNING)
        errors = sum(1 for r in reports if r.overall_result == ValidationResult.ERROR)
        failed = sum(1 for r in reports if r.overall_result == ValidationResult.FAIL)
        
        print(f"📊 Total templates: {total_templates}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Errors: {errors}")
        print(f"❌ Failed: {failed}")
        
        # Average scores
        if reports:
            avg_validation_score = sum(r.validation_score for r in reports) / len(reports)
            avg_trading_relevance = sum(r.trading_relevance_score for r in reports) / len(reports)
            avg_code_quality = sum(r.code_quality_score for r in reports) / len(reports)
        else:
            avg_validation_score = 0.0
            avg_trading_relevance = 0.0
            avg_code_quality = 0.0
        
        print(f"\n📈 Average Scores:")
        print(f"├── Validation: {avg_validation_score:.2f}")
        print(f"├── Trading relevance: {avg_trading_relevance:.2f}")
        print(f"└── Code quality: {avg_code_quality:.2f}")
        
        # Top issues
        all_issues = []
        for report in reports:
            all_issues.extend(report.issues)
        
        if all_issues:
            issue_categories = {}
            for issue in all_issues:
                if issue.category not in issue_categories:
                    issue_categories[issue.category] = 0
                issue_categories[issue.category] += 1
            
            print(f"\n🚨 Top Issue Categories:")
            for category, count in sorted(issue_categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"├── {category}: {count} issues")
        
        # Performance metrics
        if reports:
            total_validation_time = sum(r.validation_time_ms for r in reports)
            print(f"\n⏱️  Total validation time: {total_validation_time:.1f}ms")
            print(f"⚡ Average per template: {total_validation_time/len(reports):.1f}ms")
        else:
            print(f"\n⏱️  No templates to validate")
    
    def export_validation_results(self, output_dir: Path):
        """Export validation results to files"""
        output_dir.mkdir(exist_ok=True)
        
        # Export individual reports
        for report in self.validation_reports:
            report_data = asdict(report)
            report_data['timestamp'] = report_data['timestamp'].isoformat()
            report_data['validation_level'] = report_data['validation_level'].value
            report_data['overall_result'] = report_data['overall_result'].value
            
            # Convert validation issues enum values
            for issue in report_data['issues']:
                issue['severity'] = issue['severity'].value
            
            report_file = output_dir / f"validation_{report.template_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        
        # Export summary report
        summary_data = {
            'total_templates': len(self.validation_reports),
            'validation_summary': {
                'passed': sum(1 for r in self.validation_reports if r.overall_result == ValidationResult.PASS),
                'warnings': sum(1 for r in self.validation_reports if r.overall_result == ValidationResult.WARNING),
                'errors': sum(1 for r in self.validation_reports if r.overall_result == ValidationResult.ERROR),
                'failed': sum(1 for r in self.validation_reports if r.overall_result == ValidationResult.FAIL),
            },
            'average_scores': {
                'validation': sum(r.validation_score for r in self.validation_reports) / len(self.validation_reports),
                'trading_relevance': sum(r.trading_relevance_score for r in self.validation_reports) / len(self.validation_reports),
                'code_quality': sum(r.code_quality_score for r in self.validation_reports) / len(self.validation_reports),
            },
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = output_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"💾 Validation results exported to {output_dir}")

def main():
    """Main function to test workflow validation system"""
    print("🔍 ULTRATHINK Workflow Validation System")
    print("=" * 60)
    
    # Initialize template engine and generate test templates
    template_engine = WorkflowTemplateEngine()
    
    # Generate test templates
    test_workflows = [
        WorkflowType.KELLY_OPTIMIZATION,
        WorkflowType.ENSEMBLE_TRAINING,
        WorkflowType.LIVE_DEPLOYMENT,
        WorkflowType.RISK_MANAGEMENT,
        WorkflowType.STRATEGY_BACKTESTING
    ]
    
    templates = []
    for workflow_type in test_workflows:
        try:
            # Use the template generator within the engine
            template = template_engine.template_generator.generate_workflow_template(
                workflow_type=workflow_type,
                user_query=f"Generate {workflow_type.value} workflow",
                complexity_level=WorkflowComplexity.INTERMEDIATE
            )
            templates.append(template)
        except Exception as e:
            print(f"❌ Failed to generate {workflow_type.value}: {e}")
    
    # Run validation suite
    validation_suite = WorkflowValidationSuite()
    validation_reports = validation_suite.validate_all_templates(
        templates=templates,
        validation_level=ValidationLevel.COMPREHENSIVE
    )
    
    # Export results
    results_dir = Path("validation_results")
    validation_suite.export_validation_results(results_dir)
    
    print(f"\n🎯 Validation complete! Results saved to {results_dir}")

if __name__ == "__main__":
    main()