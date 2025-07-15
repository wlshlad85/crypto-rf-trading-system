#!/usr/bin/env python3
"""
Test script for production monitoring system
"""

import time
from pathlib import Path
from monitoring_alerting_system import ProductionMonitoringSystem

def main():
    print("🏥 Testing Production Monitoring System")
    print("=" * 50)
    
    # Initialize monitoring system
    config_path = Path("monitoring_config")
    monitoring_system = ProductionMonitoringSystem(config_path)
    
    # Start monitoring
    monitoring_system.start()
    
    try:
        # Let it collect some metrics
        print("📊 Collecting initial metrics...")
        time.sleep(3)
        
        # Generate health report
        health_report = monitoring_system.generate_health_report()
        print(health_report)
        
        # Get dashboard data
        dashboard_data = monitoring_system.get_dashboard_data()
        print(f"\n📈 System Health Summary:")
        print(f"├── Overall Health: {dashboard_data['overall_health']}")
        print(f"├── Performance Score: {dashboard_data['performance_score']:.1f}/100")
        print(f"├── Uptime: {dashboard_data['uptime_percentage']:.2f}%")
        print(f"├── Active Alerts: {len(dashboard_data['active_alerts'])}")
        print(f"└── Services Running: {len([s for s in dashboard_data['service_status'].values() if s == 'running'])}")
        
        print(f"\n✅ Monitoring system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during monitoring test: {e}")
        
    finally:
        # Stop monitoring
        monitoring_system.stop()
        print("🛑 Monitoring system stopped")

if __name__ == "__main__":
    main()