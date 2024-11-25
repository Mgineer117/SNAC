from safety_gymnasium.assets.geoms import Walls
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CustomWalls(Walls):
    """Custom version of the Walls class with a modified get_config method."""
    
    # You can modify the attributes here, or leave them unchanged
    num: int = 2  # Default number of walls
    locate_factor: float = 1.125
    placements: list = None  # Use if necessary
    locations: list = field(default_factory=list)  # Ensure this is initialized properly
    keepout: float = 0.0  # This can be used or ignored
    color: np.array = np.array([1.0, 0.0, 0.0, 1.0])  # Custom red color (RGBA)
    group: np.array = np.array([1, 0, 0])  # Custom group color (RGB)
    is_lidar_observed: bool = False  # Whether lidar observes the walls
    is_constrained: bool = False  # Whether the walls are constrained
    
    # Override __post_init__ if you need to modify initialization logic
    def __post_init__(self):
        assert self.num in (2, 4), 'Walls are specific for Circle and Run tasks.'
        assert self.locate_factor >= 0, 'Locate factor must be >= 0.'
        self.locations = [
            (self.locate_factor, 0),
            (-self.locate_factor, 0),
            (0, self.locate_factor),
            (0, -self.locate_factor),
        ]
    
    # Custom implementation of get_config method
    def get_config(self, xy_pos, rot):
        """Override to implement custom logic for generating the config."""
        geom = {
            'name': self.name,
            'size': np.array([0.05, 3.5, 0.3]),  # Custom size
            'pos': np.r_[xy_pos, 0.25],  # Position with custom z-coordinate
            'rot': 0,  # Default rotation
            'type': 'box',  # Type of the geometry (can be 'box', 'sphere', etc.)
            'group': self.group,  # Grouping information
            'rgba': self.color * [1, 1, 1, 0.1],  # Transparency and color
        }
        
        if self.index >= 2:  # Add some rotation based on index
            geom.update({'rot': np.pi / 2})
        
        self.index_tick()  # Update the index
        return geom
    
    def index_tick(self):
        """Increments and wraps the index."""
        self.index += 1
        self.index %= self.num