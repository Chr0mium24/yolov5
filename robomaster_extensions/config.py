"""
RoboMaster Configuration Module

Centralized configuration for RoboMaster armor plate detection system.
This module provides consistent class definitions and configuration
across all RoboMaster extensions.
"""

from typing import Dict, List, Optional, Union
import yaml
from pathlib import Path


class RoboMasterConfig:
    """Centralized configuration for RoboMaster system."""

    # Default RoboMaster class definitions (Updated for 14-class system)
    DEFAULT_CLASS_MAP = {
        'B1': 0,      # 蓝方1号
        'B2': 1,      # 蓝方2号
        'B3': 2,      # 蓝方3号
        'B4': 3,      # 蓝方4号
        'B5': 4,      # 蓝方5号
        'BHero': 5,   # 蓝方英雄
        'R1': 6,      # 红方1号
        'R2': 7,      # 红方2号
        'R3': 8,      # 红方3号
        'R4': 9,      # 红方4号
        'R5': 10,     # 红方5号
        'RHero': 11,  # 红方英雄
        'RQS': 12,    # 红方前哨站
        'BQS': 13     # 蓝方前哨站
    }

    # Default class names (index -> name mapping)
    DEFAULT_CLASS_NAMES = {
        0: 'B1',
        1: 'B2',
        2: 'B3',
        3: 'B4',
        4: 'B5',
        5: 'BHero',
        6: 'R1',
        7: 'R2',
        8: 'R3',
        9: 'R4',
        10: 'R5',
        11: 'RHero',
        12: 'RQS',
        13: 'BQS'
    }

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize RoboMaster configuration.

        Args:
            config_path: Optional path to custom configuration file
        """
        self._class_map = self.DEFAULT_CLASS_MAP.copy()
        self._class_names = self.DEFAULT_CLASS_NAMES.copy()

        if config_path:
            self.load_from_file(config_path)

    @property
    def class_map(self) -> Dict[str, int]:
        """Get class name to index mapping."""
        return self._class_map.copy()

    @property
    def class_names(self) -> Dict[int, str]:
        """Get class index to name mapping."""
        return self._class_names.copy()

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self._class_map)

    @property
    def sentry_classes(self) -> List[int]:
        """Get sentry class indices (RQS=12, BQS=13)."""
        return [self._class_map.get('RQS', 12), self._class_map.get('BQS', 13)]

    @property
    def vehicle_classes(self) -> List[int]:
        """Get vehicle class indices (all non-sentry classes)."""
        sentry_indices = set(self.sentry_classes)
        return [idx for idx in self._class_map.values() if idx not in sentry_indices]

    def get_class_index(self, class_name: str) -> int:
        """
        Get class index by name.

        Args:
            class_name: Name of the class

        Returns:
            Class index

        Raises:
            ValueError: If class name not found
        """
        if class_name not in self._class_map:
            raise ValueError(f"Class '{class_name}' not found in configuration")
        return self._class_map[class_name]

    def get_class_name(self, class_index: int) -> str:
        """
        Get class name by index.

        Args:
            class_index: Index of the class

        Returns:
            Class name

        Raises:
            ValueError: If class index not found
        """
        if class_index not in self._class_names:
            raise ValueError(f"Class index {class_index} not found in configuration")
        return self._class_names[class_index]

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if 'names' in config_data:
            # Load from YOLOv5-style config (index -> name)
            self._class_names = config_data['names']
            self._class_map = {name: idx for idx, name in self._class_names.items()}
        elif 'class_map' in config_data:
            # Load from custom format (name -> index)
            self._class_map = config_data['class_map']
            self._class_names = {idx: name for name, idx in self._class_map.items()}

    def save_to_file(self, config_path: Union[str, Path],
                     format_type: str = 'yolo') -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save configuration file
            format_type: Format type ('yolo' or 'custom')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == 'yolo':
            config_data = {
                'nc': self.num_classes,
                'names': self._class_names
            }
        else:  # custom format
            config_data = {
                'class_map': self._class_map,
                'num_classes': self.num_classes
            }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

    def is_sentry_class(self, class_index: int) -> bool:
        """Check if class index corresponds to sentry."""
        return class_index in self.sentry_classes

    def is_vehicle_class(self, class_index: int) -> bool:
        """Check if class index corresponds to a vehicle."""
        return class_index in self.vehicle_classes


# Global configuration instance
_global_config = None


def get_robomaster_config(config_path: Optional[Union[str, Path]] = None) -> RoboMasterConfig:
    """
    Get global RoboMaster configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RoboMaster configuration instance
    """
    global _global_config

    if _global_config is None or config_path is not None:
        _global_config = RoboMasterConfig(config_path)

    return _global_config


def load_config_from_yaml(yaml_path: Union[str, Path]) -> RoboMasterConfig:
    """
    Load RoboMaster configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        RoboMaster configuration instance
    """
    return RoboMasterConfig(yaml_path)


# Convenience functions for backward compatibility
def get_class_map() -> Dict[str, int]:
    """Get class name to index mapping."""
    return get_robomaster_config().class_map


def get_class_names() -> Dict[int, str]:
    """Get class index to name mapping."""
    return get_robomaster_config().class_names


def get_num_classes() -> int:
    """Get number of classes."""
    return get_robomaster_config().num_classes


def get_sentry_classes() -> List[int]:
    """Get sentry class indices."""
    return get_robomaster_config().sentry_classes


def get_vehicle_classes() -> List[int]:
    """Get vehicle class indices."""
    return get_robomaster_config().vehicle_classes