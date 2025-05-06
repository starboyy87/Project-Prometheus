"""
Performance Monitoring Dashboard for Project Prometheus
Displays real-time metrics and system performance
"""
import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import psutil
import logging
from typing import Dict, List, Any, Optional
import argparse
from prometheus_config import config

# Configure logging
log_level = config.get('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "monitor.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("monitor")

class PrometheusMonitor:
    def __init__(self, log_file: str = "performance_metrics.log", update_interval: int = 2):
        """
        Initialize the monitoring dashboard
        
        Args:
            log_file: Path to the performance metrics log file
            update_interval: Interval in seconds between dashboard updates
        """
        self.log_file = os.path.join("logs", log_file)
        self.update_interval = update_interval
        self.metrics_data = pd.DataFrame()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Project Prometheus Performance Monitor", fontsize=16)
        
        # Initialize empty plots
        self.latency_line, = self.axes[0, 0].plot([], [], 'b-', label='Query Latency (ms)')
        self.cpu_line, = self.axes[0, 1].plot([], [], 'r-', label='CPU Usage (%)')
        self.memory_line, = self.axes[1, 0].plot([], [], 'g-', label='Memory Usage (%)')
        self.disk_line, = self.axes[1, 1].plot([], [], 'y-', label='Disk Usage (%)')
        
        # Configure plots
        self.axes[0, 0].set_title('Query Latency')
        self.axes[0, 0].set_xlabel('Time')
        self.axes[0, 0].set_ylabel('Latency (ms)')
        self.axes[0, 0].legend()
        
        self.axes[0, 1].set_title('CPU Usage')
        self.axes[0, 1].set_xlabel('Time')
        self.axes[0, 1].set_ylabel('Usage (%)')
        self.axes[0, 1].legend()
        
        self.axes[1, 0].set_title('Memory Usage')
        self.axes[1, 0].set_xlabel('Time')
        self.axes[1, 0].set_ylabel('Usage (%)')
        self.axes[1, 0].legend()
        
        self.axes[1, 1].set_title('Disk Usage')
        self.axes[1, 1].set_xlabel('Time')
        self.axes[1, 1].set_ylabel('Usage (%)')
        self.axes[1, 1].legend()
        
        # Set y-axis limits
        self.axes[0, 0].set_ylim(0, 500)  # Latency up to 500ms
        self.axes[0, 1].set_ylim(0, 100)  # CPU up to 100%
        self.axes[1, 0].set_ylim(0, 100)  # Memory up to 100%
        self.axes[1, 1].set_ylim(0, 100)  # Disk up to 100%
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Initialize time data
        self.times = []
        self.latencies = []
        self.cpu_usages = []
        self.memory_usages = []
        self.disk_usages = []
        
        # Maximum number of data points to show
        self.max_data_points = 100
        
    def load_metrics(self) -> None:
        """Load metrics from the log file"""
        try:
            if os.path.exists(self.log_file):
                # Read the log file
                # Format: timestamp,function,duration_ms,cpu_usage_percent,memory_usage_percent
                df = pd.read_csv(self.log_file, header=None, 
                                names=['timestamp', 'function', 'duration_ms', 'cpu_usage_percent', 'memory_usage_percent'])
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Store the data
                self.metrics_data = df
                logger.info(f"Loaded {len(df)} metrics from {self.log_file}")
            else:
                logger.warning(f"Log file {self.log_file} does not exist")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def update_plot(self, frame: int) -> List:
        """Update the plots with new data"""
        # Load the latest metrics
        self.load_metrics()
        
        # Get current system metrics
        current_time = datetime.datetime.now()
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Calculate average latency from recent data
        if not self.metrics_data.empty:
            # Get data from the last minute
            recent_data = self.metrics_data[self.metrics_data['timestamp'] > time.time() - 60]
            if not recent_data.empty:
                avg_latency = recent_data['duration_ms'].mean()
            else:
                avg_latency = 0
        else:
            avg_latency = 0
        
        # Add new data points
        self.times.append(current_time)
        self.latencies.append(avg_latency)
        self.cpu_usages.append(cpu_percent)
        self.memory_usages.append(memory_percent)
        self.disk_usages.append(disk_percent)
        
        # Limit the number of data points
        if len(self.times) > self.max_data_points:
            self.times = self.times[-self.max_data_points:]
            self.latencies = self.latencies[-self.max_data_points:]
            self.cpu_usages = self.cpu_usages[-self.max_data_points:]
            self.memory_usages = self.memory_usages[-self.max_data_points:]
            self.disk_usages = self.disk_usages[-self.max_data_points:]
        
        # Update the plots
        self.latency_line.set_data(self.times, self.latencies)
        self.cpu_line.set_data(self.times, self.cpu_usages)
        self.memory_line.set_data(self.times, self.memory_usages)
        self.disk_line.set_data(self.times, self.disk_usages)
        
        # Adjust x-axis limits
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view(scalex=True, scaley=False)
        
        # Add threshold line for latency
        self.axes[0, 0].axhline(y=300, color='r', linestyle='--', alpha=0.7, label='Max Latency Threshold')
        
        return [self.latency_line, self.cpu_line, self.memory_line, self.disk_line]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a performance report"""
        self.load_metrics()
        
        if self.metrics_data.empty:
            return {"error": "No metrics data available"}
        
        # Calculate statistics
        avg_latency = self.metrics_data['duration_ms'].mean()
        max_latency = self.metrics_data['duration_ms'].max()
        p95_latency = self.metrics_data['duration_ms'].quantile(0.95)
        
        # Count functions exceeding thresholds
        over_max = len(self.metrics_data[self.metrics_data['duration_ms'] > 300])
        over_warning = len(self.metrics_data[self.metrics_data['duration_ms'] > 200])
        
        # Group by function
        func_stats = self.metrics_data.groupby('function').agg({
            'duration_ms': ['mean', 'max', 'count'],
            'cpu_usage_percent': 'mean',
            'memory_usage_percent': 'mean'
        })
        
        # Flatten the multi-index columns
        func_stats.columns = ['avg_latency', 'max_latency', 'call_count', 'avg_cpu', 'avg_memory']
        func_stats = func_stats.reset_index()
        
        # Sort by average latency
        func_stats = func_stats.sort_values('avg_latency', ascending=False)
        
        return {
            "overall": {
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "p95_latency_ms": p95_latency,
                "calls_over_max_threshold": over_max,
                "calls_over_warning_threshold": over_warning,
                "total_calls": len(self.metrics_data)
            },
            "by_function": func_stats.to_dict(orient='records')
        }
    
    def print_report(self) -> None:
        """Print a performance report to the console"""
        report = self.generate_report()
        
        if "error" in report:
            print(f"Error: {report['error']}")
            return
        
        print("\n" + "="*50)
        print("PROMETHEUS PERFORMANCE REPORT")
        print("="*50)
        
        overall = report["overall"]
        print(f"\nOVERALL STATISTICS:")
        print(f"Average Latency: {overall['avg_latency_ms']:.2f} ms")
        print(f"Maximum Latency: {overall['max_latency_ms']:.2f} ms")
        print(f"95th Percentile Latency: {overall['p95_latency_ms']:.2f} ms")
        print(f"Calls Exceeding Max Threshold (300ms): {overall['calls_over_max_threshold']}")
        print(f"Calls Exceeding Warning Threshold (200ms): {overall['calls_over_warning_threshold']}")
        print(f"Total Function Calls: {overall['total_calls']}")
        
        print("\nPERFORMANCE BY FUNCTION:")
        print("-"*50)
        print(f"{'Function':<20} {'Avg Latency':<15} {'Max Latency':<15} {'Call Count':<10} {'Avg CPU %':<10} {'Avg Mem %':<10}")
        print("-"*50)
        
        for func in report["by_function"]:
            print(f"{func['function']:<20} {func['avg_latency']:<15.2f} {func['max_latency']:<15.2f} {func['call_count']:<10} {func['avg_cpu']:<10.2f} {func['avg_memory']:<10.2f}")
    
    def run(self) -> None:
        """Run the monitoring dashboard"""
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=self.update_interval*1000, blit=True)
        plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prometheus Performance Monitor")
    parser.add_argument("--report", action="store_true", help="Generate a performance report instead of showing the dashboard")
    parser.add_argument("--interval", type=int, default=2, help="Update interval in seconds")
    args = parser.parse_args()
    
    monitor = PrometheusMonitor(update_interval=args.interval)
    
    if args.report:
        monitor.print_report()
    else:
        monitor.run()

if __name__ == "__main__":
    main()
