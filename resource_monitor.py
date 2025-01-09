import GPUtil
import psutil
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
import time
from datetime import datetime
import json
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# Style configuration
STYLE_CONFIG = {
    "background_color": "#1a1a2e",
    "primary_color": "#ff2a6d", 
    "secondary_color": "#05d9e8",
    "grid_color": "#ffffff",
    "text_color": "#ffffff",
    "accent_color": "#d1f5ff",
    "vram_color": "#ffd700",
    "memory_absolute_color": "#ff9966",
    "annotation_colors": ["#00ff99", "#ff66b2", "#66d9ff", "#ffcc66", "#cc99ff"],  # Cycle through these
    "font_sizes": {"title": 16, "label": 14, "tick": 12, "text": 10}
}

class ResourceMonitor(Thread):
    def __init__(self, duration=60, interval=1):
        super(ResourceMonitor, self).__init__()
        self.duration = duration
        self.interval = interval
        self.stopped = False
        self.data = {
            'timestamps': [],
            'cpu': {'usage': [], 'baseline': None},
            'ram': {'usage': [], 'absolute': [], 'baseline': None, 'baseline_abs': None, 'total': 0},
            'gpu': {'usage': [], 'baseline': None},
            'vram': {'usage': [], 'absolute': [], 'baseline': None, 'baseline_abs': None, 'total': 0}
        }
        
    def print_status(self, gpu_usage, cpu_usage, ram_data, vram_data):
        """Print current resource usage status"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\r[{timestamp}] GPU: {gpu_usage:.1f}% | "
              f"VRAM: {vram_data['usage']:.1f}% ({vram_data['used']:.1f}/{vram_data['total']:.1f} GB) | "
              f"CPU: {cpu_usage:.1f}% | "
              f"RAM: {ram_data['usage']:.1f}% ({ram_data['used']:.1f}/{ram_data['total']:.1f} GB)", end="")

    def run(self):
        start_time = time.time()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Resource Monitor...")
        print(f"Monitoring system resources for {self.duration} seconds...")
        print("\nDetected Hardware:")
        
        # Initialize hardware and get baselines
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}: {gpu.name}")
                vram_gb = gpu.memoryTotal / 1024
                print(f"    Total VRAM: {vram_gb:.2f} GB")
                self.data['vram']['total'] += vram_gb
            
            # Set GPU and VRAM baselines
            self.data['gpu']['baseline'] = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            self.data['vram']['baseline'] = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
            self.data['vram']['baseline_abs'] = sum(gpu.memoryUsed for gpu in gpus) / 1024
        
        # Set CPU and RAM baselines
        self.data['cpu']['baseline'] = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        self.data['ram']['total'] = ram.total / (1024**3)
        self.data['ram']['baseline'] = ram.percent
        self.data['ram']['baseline_abs'] = ram.used / (1024**3)
        
        print(f"CPU: {psutil.cpu_count()} cores")
        print(f"RAM: {self.data['ram']['total']:.1f} GB total")
        print("\nMonitoring in progress...")
        
        while not self.stopped and (time.time() - start_time) < self.duration:
            current_time = time.time() - start_time
            
            # Get GPU and VRAM usage
            if gpus:
                gpu_usage = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                vram_usage = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
                vram_used = sum(gpu.memoryUsed for gpu in gpus) / 1024
            else:
                gpu_usage = vram_usage = vram_used = 0
            
            # Get CPU and RAM usage
            ram = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent()
            ram_usage = ram.percent
            ram_used = ram.used / (1024**3)
            
            # Store metrics
            self.data['timestamps'].append(current_time)
            self.data['cpu']['usage'].append(cpu_usage)
            self.data['ram']['usage'].append(ram_usage)
            self.data['ram']['absolute'].append(ram_used)
            self.data['gpu']['usage'].append(gpu_usage)
            self.data['vram']['usage'].append(vram_usage)
            self.data['vram']['absolute'].append(vram_used)
            
            # Print current status
            self.print_status(
                gpu_usage, 
                cpu_usage,
                {'usage': ram_usage, 'used': ram_used, 'total': self.data['ram']['total']},
                {'usage': vram_usage, 'used': vram_used, 'total': self.data['vram']['total']}
            )
            
            time.sleep(self.interval)
            
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring completed!")
        
        # Save data to JSON
        self.save_data()
        
    def save_data(self):
        """Save monitoring data to JSON file"""
        with open('monitoring_data.json', 'w') as f:
            json.dump(self.data, f)
        print(f"Monitoring data saved to monitoring_data.json")
            
    def stop(self):
        self.stopped = True

def plot_monitoring_data(data, annotations=None):
    """Generate plots from monitoring data with optional annotations"""
    plt.style.use('dark_background')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Set background color for all subplots
    fig.patch.set_facecolor(STYLE_CONFIG["background_color"])
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(STYLE_CONFIG["background_color"])
        ax.grid(color=STYLE_CONFIG["grid_color"], alpha=0.3)
        ax.tick_params(colors=STYLE_CONFIG["text_color"], labelsize=STYLE_CONFIG["font_sizes"]["tick"])
        ax.set_ylim(0, 100)  # Set y-axis limit to 0-100 for all plots
    
    # Plot CPU usage
    timestamps = data['timestamps']
    ax1.plot(timestamps, data['cpu']['usage'], color=STYLE_CONFIG["secondary_color"], label='CPU Usage (%)')
    if data['cpu']['baseline'] is not None:
        ax1.axhline(y=data['cpu']['baseline'], color='white', linestyle='--', alpha=0.5)
    
    # Plot RAM usage
    ax2.plot(timestamps, data['ram']['usage'], color=STYLE_CONFIG["accent_color"], label='RAM Usage (%)')
    if data['ram']['baseline'] is not None:
        ax2.axhline(y=data['ram']['baseline'], color='white', linestyle='--', alpha=0.5)
    
    # Plot GPU usage
    ax3.plot(timestamps, data['gpu']['usage'], color=STYLE_CONFIG["primary_color"], label='GPU Usage (%)')
    if data['gpu']['baseline'] is not None:
        ax3.axhline(y=data['gpu']['baseline'], color='white', linestyle='--', alpha=0.5)
    
    # Plot VRAM usage
    ax4.plot(timestamps, data['vram']['usage'], color=STYLE_CONFIG["vram_color"], label='VRAM Usage (%)')
    if data['vram']['baseline'] is not None:
        ax4.axhline(y=data['vram']['baseline'], color='white', linestyle='--', alpha=0.5)
    
    # Add titles and labels
    ax1.set_title('CPU Usage', color=STYLE_CONFIG["text_color"], fontsize=STYLE_CONFIG["font_sizes"]["title"])
    ax2.set_title('RAM Usage', color=STYLE_CONFIG["text_color"], fontsize=STYLE_CONFIG["font_sizes"]["title"])
    ax3.set_title('GPU Usage', color=STYLE_CONFIG["text_color"], fontsize=STYLE_CONFIG["font_sizes"]["title"])
    ax4.set_title('VRAM Usage', color=STYLE_CONFIG["text_color"], fontsize=STYLE_CONFIG["font_sizes"]["title"])
    
    # Add annotations if provided
    if annotations:
        color_idx = 0
        for label, (start, end) in annotations.items():
            color = STYLE_CONFIG["annotation_colors"][color_idx % len(STYLE_CONFIG["annotation_colors"])]
            for ax in [ax1, ax2, ax3, ax4]:
                # Add colored rectangle at bottom of plot
                rect = Rectangle((start, 0), end - start, 2, facecolor=color, alpha=0.3)
                ax.add_patch(rect)
                
                # Add label
                text = ax.text((start + end) / 2, -5, label, ha='center', va='top', 
                             color=color, fontsize=8, rotation=45)
                text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
            
            color_idx += 1
    
    # Finalize layout
    plt.tight_layout()
    plt.savefig('resource_usage.png')
    plt.close()
    
    print(f"Plot saved as 'resource_usage.png'")

def annotate_last_run(annotations):
    """Load the last run's data and create an annotated plot"""
    try:
        with open('monitoring_data.json', 'r') as f:
            data = json.load(f)
        plot_monitoring_data(data, annotations)
    except FileNotFoundError:
        print("No monitoring data found. Please run the monitor first.")

if __name__ == "__main__":
    # Example usage:
    
    # 1. Run monitoring
    monitor = ResourceMonitor(duration=60)
    monitor.start()
    time.sleep(60)
    monitor.stop()
    plot_monitoring_data(monitor.data)  # Generate initial plot
    
    # 2. Later, add annotations (example)
    annotations = {
        "Loading Data": (5, 15),
        "Processing": (15, 30),
        "Training": (30, 45),
        "Evaluation": (45, 55)
    }
    annotate_last_run(annotations)