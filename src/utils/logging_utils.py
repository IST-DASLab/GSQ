import logging
import os
from datetime import datetime

class QuantizationLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('quantization')
        self.logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(
            os.path.join(log_dir, f'quantization_{timestamp}.log')
        )
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
    def log_config(self, config):
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
            
    def log_metrics(self, metrics, step=None):
        step_str = f" at step {step}" if step is not None else ""
        self.logger.info(f"Metrics{step_str}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")