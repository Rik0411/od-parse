"""
Simulated Model Data for Mechanical Drawing Parser

This module contains hard-coded simulation data that mimics the output of:
- YOLOv11-obb: Oriented Bounding Boxes (OBBs) for annotation detection
- Donut: Structured JSON parsing for individual annotations

The data is based on bottle_2D.png technical drawing.
"""

from typing import List, Dict, Any

# Simulated YOLOv11-obb Detection Output
# Each OBB represents a detected annotation region with its location and category
SIMULATED_YOLO_OBBS: List[Dict[str, Any]] = [
    # Main View - Dimensions
    {'id': 'm_10', 'category': 'Measure', 'x': 645, 'y': 145, 'w': 30, 'h': 25, 'rotation': 0},
    {'id': 'm_4_5', 'category': 'Measure', 'x': 550, 'y': 120, 'w': 35, 'h': 25, 'rotation': 0},
    {'id': 'm_10_vert', 'category': 'Measure', 'x': 520, 'y': 155, 'w': 25, 'h': 30, 'rotation': 90},
    {'id': 'm_36_5', 'category': 'Measure', 'x': 480, 'y': 340, 'w': 50, 'h': 30, 'rotation': 90},
    {'id': 'm_2', 'category': 'Measure', 'x': 600, 'y': 620, 'w': 25, 'h': 30, 'rotation': 90},
    
    # Main View - Radii
    {'id': 'r_R20', 'category': 'Radius', 'x': 550, 'y': 105, 'w': 45, 'h': 25, 'rotation': 0},
    {'id': 'r_2xR1_top', 'category': 'Radius', 'x': 680, 'y': 140, 'w': 50, 'h': 25, 'rotation': 0},
    {'id': 'r_2xR0_3', 'category': 'Radius', 'x': 680, 'y': 165, 'w': 55, 'h': 25, 'rotation': 0},
    {'id': 'r_R10', 'category': 'Radius', 'x': 730, 'y': 305, 'w': 45, 'h': 25, 'rotation': 0},
    {'id': 'r_R12', 'category': 'Radius', 'x': 730, 'y': 440, 'w': 45, 'h': 25, 'rotation': 0},
    {'id': 'r_R8', 'category': 'Radius', 'x': 730, 'y': 515, 'w': 40, 'h': 25, 'rotation': 0},
    {'id': 'r_R24_5', 'category': 'Radius', 'x': 600, 'y': 620, 'w': 60, 'h': 25, 'rotation': 0},
    {'id': 'r_2xR1_bottom', 'category': 'Radius', 'x': 600, 'y': 645, 'w': 50, 'h': 25, 'rotation': 0},
    
    # Main View - View Callouts
    {'id': 'callout_A', 'category': 'View', 'x': 680, 'y': 200, 'w': 30, 'h': 30, 'rotation': 0},
    {'id': 'callout_B', 'category': 'View', 'x': 680, 'y': 580, 'w': 30, 'h': 30, 'rotation': 0},
    
    # Detail View A - Dimensions
    {'id': 'm_0_15_top', 'category': 'Measure', 'x': 135, 'y': 288, 'w': 50, 'h': 25, 'rotation': 0},
    {'id': 'm_0_15_btm', 'category': 'Measure', 'x': 135, 'y': 342, 'w': 50, 'h': 25, 'rotation': 0},
    {'id': 'm_0_5_left', 'category': 'Measure', 'x': 265, 'y': 258, 'w': 40, 'h': 25, 'rotation': 0},
    {'id': 'm_0_5_right', 'category': 'Measure', 'x': 320, 'y': 258, 'w': 40, 'h': 25, 'rotation': 0},
    {'id': 'm_0_6', 'category': 'Measure', 'x': 200, 'y': 385, 'w': 40, 'h': 25, 'rotation': 0},
    {'id': 'm_1_top', 'category': 'Measure', 'x': 265, 'y': 425, 'w': 30, 'h': 25, 'rotation': 0},
    {'id': 'm_1_bottom', 'category': 'Measure', 'x': 265, 'y': 450, 'w': 30, 'h': 25, 'rotation': 0},
    
    # Detail View A - Label
    {'id': 'v_A', 'category': 'View', 'x': 240, 'y': 690, 'w': 150, 'h': 30, 'rotation': 0},
    
    # Detail View B - Label
    {'id': 'v_B', 'category': 'View', 'x': 680, 'y': 940, 'w': 150, 'h': 30, 'rotation': 0},
]

# Simulated Donut Parser Output
# Maps OBB ID to structured JSON parsing result
SIMULATED_DONUT_OUTPUTS: Dict[str, Dict[str, Any]] = {
    # Main View - Measures
    'm_10': {'type': 'LinearDimension', 'value': 10, 'unit': None, 'view': 'main_view'},
    'm_4_5': {'type': 'LinearDimension', 'value': 4.5, 'unit': None, 'view': 'main_view'},
    'm_10_vert': {'type': 'LinearDimension', 'value': 10, 'unit': None, 'view': 'main_view'},
    'm_36_5': {'type': 'LinearDimension', 'value': 36.5, 'unit': None, 'view': 'main_view'},
    'm_2': {'type': 'LinearDimension', 'value': 2, 'unit': None, 'view': 'main_view'},
    
    # Main View - Radii
    'r_R20': {'type': 'Radius', 'value': 20, 'count': 1, 'view': 'main_view'},
    'r_2xR1_top': {'type': 'Radius', 'value': 1, 'count': 2, 'view': 'main_view'},
    'r_2xR0_3': {'type': 'Radius', 'value': 0.3, 'count': 2, 'view': 'main_view'},
    'r_R10': {'type': 'Radius', 'value': 10, 'count': 1, 'view': 'main_view'},
    'r_R12': {'type': 'Radius', 'value': 12, 'count': 1, 'view': 'main_view'},
    'r_R8': {'type': 'Radius', 'value': 8, 'count': 1, 'view': 'main_view'},
    'r_R24_5': {'type': 'Radius', 'value': 24.5, 'count': 1, 'view': 'main_view'},
    'r_2xR1_bottom': {'type': 'Radius', 'value': 1, 'count': 2, 'view': 'main_view'},
    
    # Main View - View Callouts
    'callout_A': {'type': 'ViewCallout', 'label': 'A', 'target_view': 'detail_view_a', 'view': 'main_view'},
    'callout_B': {'type': 'ViewCallout', 'label': 'B', 'target_view': 'detail_view_b', 'view': 'main_view'},
    
    # Detail View A - Measures
    'm_0_15_top': {'type': 'LinearDimension', 'value': 0.15, 'unit': None, 'view': 'detail_view_a'},
    'm_0_15_btm': {'type': 'LinearDimension', 'value': 0.15, 'unit': None, 'view': 'detail_view_a'},
    'm_0_5_left': {'type': 'LinearDimension', 'value': 0.5, 'unit': None, 'view': 'detail_view_a'},
    'm_0_5_right': {'type': 'LinearDimension', 'value': 0.5, 'unit': None, 'view': 'detail_view_a'},
    'm_0_6': {'type': 'LinearDimension', 'value': 0.6, 'unit': None, 'view': 'detail_view_a'},
    'm_1_top': {'type': 'LinearDimension', 'value': 1, 'unit': None, 'view': 'detail_view_a'},
    'm_1_bottom': {'type': 'LinearDimension', 'value': 1, 'unit': None, 'view': 'detail_view_a'},
    
    # View Labels
    'v_A': {'type': 'ViewLabel', 'name': 'Detail View A', 'view': 'detail_view_a'},
    'v_B': {'type': 'ViewLabel', 'name': 'Detail View B', 'view': 'detail_view_b'},
}

