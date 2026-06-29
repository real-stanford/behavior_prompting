import multiprocessing
from multiprocessing import Pool, Manager
import numpy as np
import click
from behavior_prompting.train_network.env.draw.draw_env import DrawEnv
import pygame
import os
import shutil
import time
from tqdm import tqdm
import hashlib

# Import shared functions from demo_draw.py
from demo_draw import (
    get_or_create_replay_buffer_for_task,
    get_all_task_names,
    cleanup_empty_tasks,
    get_total_episodes
)

# Import additional functions needed for video generation
from behavior_prompting.common.replay_buffer_util import print_replay_buffer_draw
from demo_draw import get_task_dataset_path

# Import group_demos function
from group_demos import group_demos
import warnings
warnings.filterwarnings("ignore", message="missing object_codec for object array", category=FutureWarning) # we get warnings which interfere with the progress bar

def seed_from_ints(ints):
    # Convert list of ints to bytes
    data = ",".join(map(str, ints)).encode()
    # Hash to get a fixed-length digest
    digest = hashlib.sha256(data).hexdigest()
    # Turn hash into integer (for seeding)
    return int(digest, 16) % (2**32)  # limit to 32-bit for RNG seeds

def sample_random_board_position(board_length, margin):
    """
    Sample a random position within the board boundaries.
    
    Args:
        board_length: Length of the drawing board (square)
        margin: Margin from board edge in pixels
    
    Returns:
        position: Random (x, y) position within board boundaries
    """
    board_center = 256.0
    half_board = board_length / 2
    
    return np.array([
        np.random.uniform(board_center - half_board + margin, board_center + half_board - margin),
        np.random.uniform(board_center - half_board + margin, board_center + half_board - margin)
    ])

def sample_random_board_position_with_min_distance(start_pos, board_length, margin, min_distance):
    """
    Sample a random position within the board boundaries that is at least min_distance away from start_pos.
    
    Args:
        start_pos: Starting position (x, y) to measure distance from
        board_length: Length of the drawing board (square)
        margin: Margin from board edge in pixels
        min_distance: Minimum distance required from start_pos in pixels
    
    Returns:
        position: Random (x, y) position within board boundaries and min_distance away from start_pos
    """
    board_center = 256.0
    half_board = board_length / 2
    
    max_attempts = 100  # Prevent infinite loops
    for attempt in range(max_attempts):
        position = np.array([
            np.random.uniform(board_center - half_board + margin, board_center + half_board - margin),
            np.random.uniform(board_center - half_board + margin, board_center + half_board - margin)
        ])
        
        # Check if distance is sufficient
        if np.linalg.norm(position - start_pos) >= min_distance:
            return position
    
    # If we couldn't find a position with sufficient distance, return the original position
    # This should rarely happen unless min_distance is very large relative to board size
    return start_pos

def generate_movement_control_points(start_pos, end_pos, board_length, margin, offset_magnitude_factor=0.3, offset_magnitude_max=50.0):
    """
    Generate control points for movement parts with perpendicular offsets for smooth curves.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: End position (x, y)
        board_length: Length of the drawing board (square)
        margin: Margin from board edge in pixels
        offset_magnitude_factor: Factor to multiply distance by for offset magnitude (default: 0.3)
        offset_magnitude_max: Maximum offset magnitude in pixels (default: 50.0)
    
    Returns:
        control1, control2: Two control points for smooth Bezier curve movement
    """
    # Calculate direction and distance
    direction = end_pos - start_pos
    distance = np.linalg.norm(direction)
    
    if distance > 0:
        # Create control points with perpendicular offset for smooth curves
        perpendicular = np.array([-direction[1], direction[0]]) / distance
        offset_magnitude = min(distance * offset_magnitude_factor, offset_magnitude_max)
        
        # Control points at 1/3 and 2/3 along the path with perpendicular offset
        control1 = start_pos + direction * 0.33 + perpendicular * np.random.uniform(-offset_magnitude, offset_magnitude)
        control2 = start_pos + direction * 0.67 + perpendicular * np.random.uniform(-offset_magnitude, offset_magnitude)
        
        # Ensure control points stay within board boundaries
        board_center = 256.0
        half_board = board_length / 2
        control1 = np.clip(control1, board_center - half_board + margin, board_center + half_board - margin)
        control2 = np.clip(control2, board_center - half_board + margin, board_center + half_board - margin)
    else:
        # If start and end are the same, use the same position for control points
        control1 = start_pos
        control2 = start_pos
    
    return control1, control2

def compute_bezier_point(start_pos, control1, control2, end_pos, t):
    """
    Compute a point on a cubic Bezier curve.
    
    Args:
        start_pos: Starting position (x, y)
        control1: First control point (x, y)
        control2: Second control point (x, y)
        end_pos: End position (x, y)
        t: Parameter t in [0, 1] along the curve
    
    Returns:
        point: Position (x, y) on the Bezier curve at parameter t
    """
    t_inv = 1 - t
    
    # Cubic Bezier curve: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    x = (t_inv**3 * start_pos[0] + 
         3 * t_inv**2 * t * control1[0] + 
         3 * t_inv * t**2 * control2[0] + 
         t**3 * end_pos[0])
    y = (t_inv**3 * start_pos[1] + 
         3 * t_inv**2 * t * control1[1] + 
         3 * t_inv * t**2 * control2[1] + 
         t**3 * end_pos[1])
    
    return np.array([x, y])

def generate_straight_line_part(start_pos, end_pos):
    """
    Generate a straight line part from start_pos to end_pos.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: End position (x, y)
    
    Returns:
        control_points: Dictionary with start_pos, end_pos, pen_down, and part_type
    """
    # Return control points for speed-independent trajectory
    control_points = {
        'start_pos': start_pos,
        'end_pos': end_pos,
        'pen_down': 1.0,  # Pen down for drawing
        'part_type': 'straight'
    }
    
    return control_points

def generate_curve_part(start_pos, end_pos, pen_down=1.0, board_length=None, margin=None):
    """
    Generate a curve part using a Bezier curve from start_pos to end_pos.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: End position (x, y)
        pen_down: Whether the pen is down (1.0) or up (0.0) for this part
        board_length: Length of the drawing board (for sampling control points)
        margin: Margin from board edge (for sampling control points)
    
    Returns:
        control_points: Dictionary with start_pos, control1, control2, end_pos, pen_down, and part_type
    """
    # Sample random control points within the board boundaries
    control1 = sample_random_board_position(board_length, margin)
    control2 = sample_random_board_position(board_length, margin)
    
    # Part type is always 'curve' regardless of pen state
    # The pen state determines whether it's a drawing curve or movement curve
    part_type = 'curve'
    
    # Return control points for speed-independent trajectory
    control_points = {
        'start_pos': start_pos,
        'control1': control1,
        'control2': control2,
        'end_pos': end_pos,
        'pen_down': pen_down,
        'part_type': part_type
    }
    
    return control_points

def generate_movement_part(start_pos, end_pos, board_length=None, margin=None):
    """
    Generate a movement part (pen up) from start_pos to end_pos.
    
    Args:
        start_pos: Starting position (x, y)
        end_pos: End position (x, y)
        board_length: Length of the drawing board (for sampling control points)
        margin: Margin from board edge (for sampling control points)
    
    Returns:
        control_points: Dictionary with start_pos, control1, control2, end_pos, pen_down, and part_type
    """
    # Return control points for speed-independent trajectory
    control_points = {
        'start_pos': start_pos,
        'end_pos': end_pos,
        'pen_down': 0.0,  # Pen up for movement
        'part_type': 'movement'
    }
    
    return control_points

def check_oval_portion_fits_board(start_pos, major_axis, minor_axis, major_angle, start_angle, end_angle, board_length, margin):
    """
    Check if the drawn portion of an oval fits within the board boundaries.
    
    Args:
        start_pos: Start position of the oval drawing
        major_axis: Length of major axis
        minor_axis: Length of minor axis  
        major_angle: Orientation angle of major axis
        start_angle: Starting angle of the drawn portion (in ellipse parameter space)
        end_angle: Ending angle of the drawn portion (in ellipse parameter space)
        board_length: Length of the drawing board
        margin: Margin from board edge
    
    Returns:
        bool: True if the drawn oval portion fits within board boundaries, False otherwise
    """
    # Calculate the center of the oval such that start_pos corresponds to the start_angle
    # First, calculate where start_angle would be in the standard ellipse
    x_start_ellipse = major_axis * np.cos(start_angle)
    y_start_ellipse = minor_axis * np.sin(start_angle)
    
    # Apply rotation to get the rotated position
    cos_rot = np.cos(major_angle)
    sin_rot = np.sin(major_angle)
    x_start_rotated = x_start_ellipse * cos_rot - y_start_ellipse * sin_rot
    y_start_rotated = x_start_ellipse * sin_rot + y_start_ellipse * cos_rot
    
    # Calculate center such that start_pos = center + rotated_start_position
    center = start_pos - np.array([x_start_rotated, y_start_rotated])
    
    # Calculate board boundaries using board center
    board_center = 256.0
    half_board = board_length / 2
    min_x = board_center - half_board + margin
    max_x = board_center + half_board - margin
    min_y = board_center - half_board + margin
    max_y = board_center + half_board - margin
    
    # Check 12 equally spaced points along the drawn portion of the oval
    num_check_points = 12
    angular_range = end_angle - start_angle
    
    for i in range(num_check_points):
        # Calculate angle for this check point
        t = i / (num_check_points - 1) if num_check_points > 1 else 0
        angle = start_angle + t * angular_range
        
        # Parametric ellipse equations
        x_ellipse = major_axis * np.cos(angle)
        y_ellipse = minor_axis * np.sin(angle)
        
        # Apply rotation
        x_rotated = x_ellipse * cos_rot - y_ellipse * sin_rot
        y_rotated = x_ellipse * sin_rot + y_ellipse * cos_rot
        
        # Translate to board coordinates
        x = center[0] + x_rotated
        y = center[1] + y_rotated
        
        # Check if this point is within board boundaries
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
    
    return True

def check_oval_fits_board(start_pos, major_axis, minor_axis, major_angle, board_length, margin):
    """
    Check if a full oval with given parameters will fit within the board boundaries.
    This is a convenience wrapper for the full oval case.
    
    Args:
        start_pos: Start position (one end of major axis)
        major_axis: Length of major axis
        minor_axis: Length of minor axis  
        major_angle: Orientation angle of major axis
        board_length: Length of the drawing board
        margin: Margin from board edge
    
    Returns:
        bool: True if oval fits within board boundaries, False otherwise
    """
    return check_oval_portion_fits_board(start_pos, major_axis, minor_axis, major_angle, 
                                       0, 2 * np.pi, board_length, margin)

def generate_oval_part(start_pos, board_length=None, margin=None, major_axis_min=30.0, major_axis_max=80.0, partial_oval_probability=0.0):
    """
    Generate an oval part that starts at start_pos.
    
    Args:
        start_pos: Start position (x, y) - where the oval drawing begins
        board_length: Length of the drawing board (for sampling control points)
        margin: Margin from board edge (for sampling control points)
        major_axis_min: Minimum major axis size in pixels
        major_axis_max: Maximum major axis size in pixels
        partial_oval_probability: Probability of drawing a partial oval instead of full oval
    
    Returns:
        control_points: Dictionary with start_pos, end_pos, pen_down, part_type, and oval parameters
    """
    # Decide whether to draw a partial or full oval
    is_partial_oval = np.random.random() < partial_oval_probability
    
    # Decide direction: 50% chance for clockwise vs counterclockwise
    is_clockwise = np.random.random() < 0.5
    
    # Keep resampling until we get an oval that fits within the board
    max_attempts = 100
    for attempt in range(max_attempts):
        # Randomly sample major axis size from the specified range
        major_axis = np.random.uniform(major_axis_min, major_axis_max)
        
        # Minor axis can be any size relative to major axis
        # This allows for all types of ovals: elongated, circular, wide, etc.
        # Range from 0.2x to 3.0x allows for very thin to very wide ovals
        minor_axis = major_axis * np.random.uniform(0.2, 3.0)
        
        # For ovals, we can use any orientation
        # Let's use a random orientation for variety
        major_angle = np.random.uniform(0, 2 * np.pi)
        
        # For partial ovals, determine the angular range first
        if is_partial_oval:
            # Sample a random proportion between π/8 and 2π for partial ovals
            # Minimum of π/8 to ensure meaningful arcs
            min_proportion = np.pi / 8
            max_proportion = 2 * np.pi
            angular_range = np.random.uniform(min_proportion, max_proportion)
            
            # Always start at angle 0 so the drawing begins at start_pos
            start_angle = 0
            if is_clockwise:
                # For clockwise, go from 0 to -angular_range (negative direction)
                end_angle = start_angle - angular_range
                # Store the signed angular range for correct action generation
                angular_range = -angular_range
            else:
                # For counterclockwise, go from 0 to +angular_range (positive direction)
                end_angle = start_angle + angular_range
                # angular_range remains positive
            
            # Check if this partial oval portion will fit within the board boundaries
            if check_oval_portion_fits_board(start_pos, major_axis, minor_axis, major_angle, 
                                           start_angle, end_angle, board_length, margin):
                # Calculate actual end position for partial ovals
                # First get the center position
                x_start_ellipse = major_axis * np.cos(start_angle)
                y_start_ellipse = minor_axis * np.sin(start_angle)
                
                cos_rot = np.cos(major_angle)
                sin_rot = np.sin(major_angle)
                x_start_rotated = x_start_ellipse * cos_rot - y_start_ellipse * sin_rot
                y_start_rotated = x_start_ellipse * sin_rot + y_start_ellipse * cos_rot
                
                center = start_pos - np.array([x_start_rotated, y_start_rotated])
                
                # Calculate end position
                x_end_ellipse = major_axis * np.cos(end_angle)
                y_end_ellipse = minor_axis * np.sin(end_angle)
                x_end_rotated = x_end_ellipse * cos_rot - y_end_ellipse * sin_rot
                y_end_rotated = x_end_ellipse * sin_rot + y_end_ellipse * cos_rot
                
                end_pos = center + np.array([x_end_rotated, y_end_rotated])
                break
        else:
            # For full ovals, use the original check (which now calls the new function)
            if check_oval_fits_board(start_pos, major_axis, minor_axis, major_angle, board_length, margin):
                # For full ovals, we need to adjust so start_pos is at angle 0
                start_angle = 0
                if is_clockwise:
                    # For clockwise full ovals, go from 0 to -2π
                    end_angle = -2 * np.pi
                    angular_range = -2 * np.pi
                else:
                    # For counterclockwise full ovals, go from 0 to +2π
                    end_angle = 2 * np.pi
                    angular_range = 2 * np.pi
                end_pos = start_pos  # Same as start_pos for closed ovals
                break
    else:
        # If we couldn't find a fitting oval after max_attempts, raise an exception
        # This will trigger the fallback to a straight line in the calling function
        raise ValueError(f"Could not find valid oval parameters after {max_attempts} attempts")
    
    # Return control points for speed-independent trajectory
    control_points = {
        'start_pos': start_pos,
        'control1': None,  # Not used for ovals
        'control2': None,  # Not used for ovals
        'end_pos': end_pos,
        'pen_down': 1.0,  # Pen down for drawing
        'part_type': 'oval',
        'major_axis': major_axis,
        'minor_axis': minor_axis,
        'major_angle': major_angle,
        'is_partial': is_partial_oval,
        'start_angle': start_angle,
        'angular_range': angular_range  # This encodes both direction and magnitude
    }
    
    return control_points

def generate_procedural_trajectory(control_hz, board_length, margin, min_parts, max_parts, connection_probability=0.3, min_distance=50.0, oval_major_axis_min=30.0, oval_major_axis_max=80.0, partial_oval_probability=0.0, allowed_parts=None, movement_allowed_at_end_of_episode=True, verbose=False):
    """
    Generate procedural drawing trajectory with multiple parts (upright board).
    
    Args:
        control_hz: Control frequency in Hz
        board_length: Length of the drawing board (square)
        margin: Margin from board edge in pixels
        min_parts: Minimum number of parts in each trajectory
        max_parts: Maximum number of parts in each trajectory
        connection_probability: Probability that a new part connects to a previous part endpoint
        min_distance: Minimum distance between start and end positions for each part (default: 50.0)
        oval_major_axis_min: Minimum major axis size for oval parts in pixels (default: 30.0)
        oval_major_axis_max: Maximum major axis size for oval parts in pixels (default: 80.0)
        partial_oval_probability: Probability of drawing partial ovals instead of full ovals (default: 0.0)
        allowed_parts: List of allowed part types ['straight', 'curve', 'oval', 'movement'] or None for all (default: None)
        movement_allowed_at_end_of_episode: Whether to allow movement at the end of the episode (default: True)
        verbose: Whether to print debug information about connections
    
    Returns:
        control_points: List of control point dictionaries for each part
        start_pos: The starting position (upright)
    """
    # Sample number of parts from the specified range
    num_parts = np.random.randint(min_parts, max_parts + 1)
    
    # Start from a random position within the board boundaries
    current_pos = sample_random_board_position(board_length, margin)
    
    all_control_points = []
    previous_endpoints = []  # Track all previous endpoints for potential connections
    
    # Set up allowed part types
    if allowed_parts is None:
        allowed_parts = ['straight', 'curve', 'oval', 'movement']
    else:
        # Validate allowed_parts
        valid_parts = ['straight', 'curve', 'oval', 'movement']
        for part in allowed_parts:
            if part not in valid_parts:
                raise ValueError(f"Invalid part type '{part}'. Valid types are: {valid_parts}")
    
    # Generate parts
    for part_idx in range(num_parts):
        # For the first part, ensure it's always a drawing action (never movement)
        # For the last part, ensure it's always a drawing action (never movement) if movement is not allowed at the end of the episode
        if part_idx == 0 or (part_idx == num_parts - 1 and not movement_allowed_at_end_of_episode):
            drawing_parts = [p for p in allowed_parts if p != 'movement']
            if not drawing_parts:
                raise ValueError("At least one drawing part type (straight, curve, oval) must be allowed for the first part")
            part_type = np.random.choice(drawing_parts)
        else:
            # Choose part type randomly for middle parts, but avoid consecutive movements
            if all_control_points and all_control_points[-1]['part_type'] == 'movement':
                # If previous part was movement, force this to be drawing
                drawing_parts = [p for p in allowed_parts if p != 'movement']
                if not drawing_parts:
                    raise ValueError("At least one drawing part type (straight, curve, oval) must be allowed after movement parts")
                part_type = np.random.choice(drawing_parts)
            else:
                # Can choose any type if previous wasn't movement
                part_type = np.random.choice(allowed_parts)
        
        # Decide whether to connect to a previous endpoint or generate a new random position
        should_connect = (part_idx > 0 and  # Don't connect the first part
                         len(previous_endpoints) > 0 and  # Need previous endpoints to connect to
                         np.random.random() < connection_probability) # Random chance based on probability
        
        if should_connect:
            # Choose a random previous endpoint to connect to
            end_pos = previous_endpoints[np.random.randint(len(previous_endpoints))]
            if verbose:
                print(f"    Part {part_idx}: Connecting to previous endpoint at {end_pos}")
        else:
            # Generate a new random position with minimum distance constraint
            end_pos = sample_random_board_position_with_min_distance(current_pos, board_length, margin, min_distance)
            if verbose:
                print(f"    Part {part_idx}: Generated new position at {end_pos} (distance: {np.linalg.norm(end_pos - current_pos):.1f})")
        
        if part_type == 'straight':
            control_points = generate_straight_line_part(current_pos, end_pos)
        elif part_type == 'curve':
            control_points = generate_curve_part(current_pos, end_pos, board_length=board_length, margin=margin)
        elif part_type == 'oval':
            # Try to generate an oval, but if it fails, fall back to a straight line
            try:
                control_points = generate_oval_part(current_pos, board_length=board_length, margin=margin,
                                                 major_axis_min=oval_major_axis_min, major_axis_max=oval_major_axis_max,
                                                 partial_oval_probability=partial_oval_probability)
            except Exception:
                # If oval generation fails, generate a straight line instead
                if verbose:
                    print(f"    Part {part_idx}: Oval generation failed, falling back to straight line")
                control_points = generate_straight_line_part(current_pos, end_pos)
        else:  # movement
            control_points = generate_movement_part(current_pos, end_pos, board_length=board_length, margin=margin)
        
        all_control_points.append(control_points)
        # Use the actual end position from the generated control points
        # This is important for ovals which generate their own end_pos
        actual_end_pos = control_points['end_pos']
        
        # Sanity check: for straight and curve parts, ensure they end where expected
        if part_type in ['straight', 'curve', 'movement']:
            distance_to_expected = np.linalg.norm(actual_end_pos - end_pos)
            assert distance_to_expected <= 10.0, f"Part {part_idx} ({part_type}): actual end position {actual_end_pos} is {distance_to_expected:.2f} pixels away from expected end position {end_pos}"
        
        current_pos = actual_end_pos
        
        # Store this endpoint for potential future connections
        previous_endpoints.append(actual_end_pos)
    
    # Return control points and starting position (no rotation applied)
    start_pos = np.array([all_control_points[0]['start_pos'][0], all_control_points[0]['start_pos'][1]])
    
    return all_control_points, start_pos

def convert_control_points_to_actions(control_points, control_hz, speed, noise_std=1.0, noise_bounds=20.0, board_length=None, margin=None,
                                    offset_magnitude_factor=0.3, offset_magnitude_max=50.0,
                                    part_delay_min=0, part_delay_max=10, min_distance=50.0,
                                    final_hold_steps=10):
    """
    Convert control points to actions at a specific speed.
    
    Args:
        control_points: List of control point dictionaries
        control_hz: Control frequency in Hz
        speed: Speed in pixels per second
        noise_std: Standard deviation of Gaussian noise to add to positions (in pixels)
        noise_bounds: Maximum absolute value for noise clipping (in pixels). Set to None to disable clipping.
        board_length: Length of the drawing board (for regenerating movement control points)
        margin: Margin from board edge (for regenerating movement control points)
        offset_magnitude_factor: Factor to multiply distance by for offset magnitude
        offset_magnitude_max: Maximum offset magnitude in pixels
        part_delay_min: Minimum number of delay steps between parts (default: 0)
        part_delay_max: Maximum number of delay steps between parts (default: 10)
        min_distance: Minimum distance between start and end positions for each trajectory part in pixels (default: 50.0)
        final_hold_steps: Steps to hold at the final end position (default: 10)
    
    Returns:
        actions: Array of (x, y, pen_down) actions
        total_steps: Total number of steps
    """
    all_actions = []
    part_final_action_indices = []  # Track indices of final actions for each part
    
    # Check if the last part is a movement and modify its end position to be a new position (since it won't impact the drawing)
    if len(control_points) > 0 and control_points[-1]['part_type'] == 'movement':
        last_part = control_points[-1]
        start_pos = last_part['start_pos']
        # Sample a random end position with minimum distance constraint
        new_end_pos = sample_random_board_position_with_min_distance(start_pos, board_length, margin, min_distance)
        # Update the control point with the new end position
        control_points[-1]['end_pos'] = new_end_pos
    
    for part_idx, part in enumerate(control_points):
        part_type = part['part_type']
        start_pos = part['start_pos']
        end_pos = part['end_pos']
        pen_down = part['pen_down']
        
        if part_type == 'straight':
            # Calculate distance and duration for straight line
            distance = np.linalg.norm(end_pos - start_pos)
            duration = distance / speed
            steps = int(duration * control_hz)
            steps = max(steps, 5)  # Minimum 5 steps
            
            # Generate actions
            for step_idx in range(steps):
                t = step_idx / (steps - 1) if steps > 1 else 0
                x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                all_actions.append(np.array([x, y, pen_down], dtype=np.float32))
                
        elif part_type == 'curve':
            # Use precomputed control points for drawing curves (consistent across demos)
            control1 = part['control1']
            control2 = part['control2']
            distance = np.linalg.norm(end_pos - start_pos) * 1.5  # Curve is typically longer
            duration = distance / speed
            steps = int(duration * control_hz)
            steps = max(steps, 8)  # Minimum 8 steps
            
            # Generate actions using Bezier curve
            for step_idx in range(steps):
                t = step_idx / (steps - 1) if steps > 1 else 0
                
                # Use helper function for Bezier curve computation
                point = compute_bezier_point(start_pos, control1, control2, end_pos, t)
                
                all_actions.append(np.array([point[0], point[1], pen_down], dtype=np.float32))
                
        elif part_type == 'oval':
            # Extract oval parameters from the control points
            major_axis = part['major_axis']
            minor_axis = part['minor_axis']
            major_angle = part['major_angle']
            is_partial = part.get('is_partial', False)
            start_angle = part.get('start_angle', 0)
            angular_range = part.get('angular_range', 2 * np.pi)
            
            # Calculate distance and duration for oval
            if is_partial:
                # For partial ovals, calculate arc length
                # Use Ramanujan's approximation for ellipse perimeter, scaled by angular proportion
                h = ((major_axis - minor_axis) / (major_axis + minor_axis)) ** 2
                full_perimeter = np.pi * (major_axis + minor_axis) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
                perimeter = full_perimeter * (abs(angular_range) / (2 * np.pi))
            else:
                # Full oval perimeter
                h = ((major_axis - minor_axis) / (major_axis + minor_axis)) ** 2
                perimeter = np.pi * (major_axis + minor_axis) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
            
            duration = perimeter / speed
            steps = int(duration * control_hz)
            steps = max(steps, 8 if is_partial else 16)  # Minimum steps for smooth drawing
            
            # Generate actions using parametric ellipse equations
            for step_idx in range(steps):
                # Parameter t interpolates between start_angle and end_angle
                if steps > 1:
                    # Calculate the actual end_angle based on angular_range direction
                    # angular_range can be positive (counterclockwise) or negative (clockwise)
                    t = start_angle + (step_idx / (steps - 1)) * angular_range
                else:
                    t = start_angle
                
                # Parametric ellipse equations: x = a*cos(t), y = b*sin(t)
                x_ellipse = major_axis * np.cos(t)
                y_ellipse = minor_axis * np.sin(t)
                
                # Apply rotation to the ellipse
                cos_rot = np.cos(major_angle)
                sin_rot = np.sin(major_angle)
                x_rotated = x_ellipse * cos_rot - y_ellipse * sin_rot
                y_rotated = x_ellipse * sin_rot + y_ellipse * cos_rot
                
                # Calculate center such that start_pos corresponds to start_angle
                # First, calculate where start_angle would be in the standard ellipse
                x_start_ellipse = major_axis * np.cos(start_angle)
                y_start_ellipse = minor_axis * np.sin(start_angle)
                
                # Apply rotation to get the rotated position
                x_start_rotated = x_start_ellipse * cos_rot - y_start_ellipse * sin_rot
                y_start_rotated = x_start_ellipse * sin_rot + y_start_ellipse * cos_rot
                
                # Calculate center such that start_pos = center + rotated_start_position
                center = start_pos - np.array([x_start_rotated, y_start_rotated])
                
                x = center[0] + x_rotated
                y = center[1] + y_rotated
                
                all_actions.append(np.array([x, y, pen_down], dtype=np.float32))
                
        elif part_type == 'movement':
            # Regenerate control points for movement parts each time (adds variety)
            # Generate new random control points for this movement execution
            control1, control2 = generate_movement_control_points(start_pos, end_pos, board_length, margin, 
                                                               offset_magnitude_factor, offset_magnitude_max)
            
            # Calculate distance and duration for movement using Bezier curve
            distance = np.linalg.norm(end_pos - start_pos) * 1.5  # Curve is typically longer
            duration = distance / speed
            steps = int(duration * control_hz)
            steps = max(steps, 5)  # Minimum 5 steps for smooth movement
            
            # Generate actions using Bezier curve
            for step_idx in range(steps):
                t = step_idx / (steps - 1) if steps > 1 else 0
                
                # Use helper function for Bezier curve computation
                point = compute_bezier_point(start_pos, control1, control2, end_pos, t)
                
                all_actions.append(np.array([point[0], point[1], pen_down], dtype=np.float32))
        
        # Sanity check: ensure end_pos is close to the last action position
        if len(all_actions) > 0:
            last_action = all_actions[-1]
            distance_to_end = np.linalg.norm(np.array([end_pos[0], end_pos[1]]) - np.array([last_action[0], last_action[1]]))
            assert distance_to_end <= 10.0, f"End position {end_pos} is {distance_to_end:.2f} pixels away from last action position [{last_action[0]:.2f}, {last_action[1]:.2f}] (part_type: {part_type})"
            
            # Track the index of the final action for this part (before any delay actions)
            part_final_action_indices.append(len(all_actions) - 1)
        
        # Add delay between parts (except after the last part)
        if part_idx < len(control_points) - 1:
            delay_steps = np.random.randint(part_delay_min, part_delay_max + 1)
            if delay_steps > 0:
                # Repeat the end position action for the delay duration
                for _ in range(delay_steps):
                    all_actions.append(np.array([end_pos[0], end_pos[1], pen_down], dtype=np.float32))
    
    # Add steps of repeating the final end action at the end of the trajectory
    if len(control_points) > 0 and final_hold_steps > 0:
        final_end_pos = control_points[-1]['end_pos']
        final_pen_down = control_points[-1]['pen_down']
        for _ in range(final_hold_steps):
            all_actions.append(np.array([final_end_pos[0], final_end_pos[1], final_pen_down], dtype=np.float32))
    
    # Convert to numpy array for easier manipulation
    all_actions = np.array(all_actions)
    
    # Apply noise to all actions except the final action of each part
    if noise_std > 0 and len(all_actions) > 0:
        # Create a boolean mask for indices that should have noise applied
        noise_mask = np.ones(len(all_actions), dtype=bool)
        if part_final_action_indices:
            noise_mask[part_final_action_indices] = False
        
        # Count how many actions need noise
        num_noise_actions = np.sum(noise_mask)
        
        if num_noise_actions > 0:
            # Generate all noise at once using vectorized operations
            noise = np.random.normal(0, noise_std, size=(num_noise_actions, 2))
            
            # Apply noise bounds (clipping) if specified
            if noise_bounds is not None:
                noise = np.clip(noise, -noise_bounds, noise_bounds)
            
            # Add noise to x and y coordinates (leave pen_down unchanged) using boolean indexing
            all_actions[noise_mask, :2] += noise
    
    return all_actions, len(all_actions)

def generate_single_task(args):
    """
    Generate a single task with all its demos. This function is designed to run in a separate process.
    
    Args:
        args: Tuple containing all the arguments needed for task generation
    
    Returns:
        tuple: (task_name, episode_count)
    """
    # Unpack arguments
    (output, task_idx, num_tasks, control_hz, board_length, margin, demos_per_task,
     trajectory_speed_min, trajectory_speed_max,
     noise_std, noise_bounds, min_parts, max_parts, visualize, viewer, verbose, base_seed,
     positioning_steps_min, positioning_steps_max, offset_magnitude_factor, offset_magnitude_max, reward_threshold, connection_probability,
     part_delay_min, part_delay_max, min_distance, oval_major_axis_min, oval_major_axis_max, partial_oval_probability, allowed_parts) = args

    # Set numpy random seed for this task to ensure different random variations across processes
    task_seed = seed_from_ints([base_seed, task_idx, num_tasks])
    np.random.seed(task_seed)
    
    # Generate task name with index and leading zeros
    # Calculate the number of digits needed based on total tasks
    num_digits = len(str(num_tasks))
    task_number = str(task_idx + 1).zfill(num_digits)
    task_name = f'draw procedural_{task_number}'
    
    if verbose:
        print(f"\n--- Generating Task: {task_name} ---")
    
    # Create replay buffer for this task
    current_replay_buffer = get_or_create_replay_buffer_for_task(output, task_name, in_memory=True)
    
    # Create a single environment for this task (each process gets its own env)
    render_mode = 'human' if viewer else 'rgb_array'
    env = DrawEnv(boundary_angle=0, render_mode=render_mode)
    
    # Generate the procedural trajectory for this task (upright) - same for all demos
    trajectory_control_points, start_pos_upright = generate_procedural_trajectory(
        control_hz, board_length, margin, min_parts, max_parts, 
        trajectory_speed_min=trajectory_speed_min, trajectory_speed_max=trajectory_speed_max,
        connection_probability=connection_probability, min_distance=min_distance, 
        oval_major_axis_min=oval_major_axis_min, oval_major_axis_max=oval_major_axis_max, 
        partial_oval_probability=partial_oval_probability, allowed_parts=allowed_parts, verbose=verbose)
    
    if verbose:
        print(f"    Generated trajectory with {len(trajectory_control_points)} parts")
    
    episode_count = 0
    first_demo_drawing_image = None
    first_demo_boundary_angle = None
    
    # Generate demos for this task
    for demo_idx in range(demos_per_task):
        demo_success = False
        retry_count = 0
        max_retries = 10  # Prevent infinite loops
        
        while not demo_success and retry_count < max_retries:
            # Set numpy random seed for this process to ensure different random variations
            # Increment seed on each retry to get different drawing variations
            process_seed = seed_from_ints([base_seed, task_idx, num_tasks, demo_idx, retry_count])
            np.random.seed(process_seed)
            
            if retry_count > 0:
                print(f"    Retry {retry_count} for demo {demo_idx+1} with seed {process_seed}")

            episode_data = {
                'image': [],
                'agent_pos': [], 
                'pen_down': [],
                'action': []
            }
            episode_labels = {
                'boundary_angle': [],
                'drawing_image': []
            }
            
            # Set rotation for this demo (random rotation between -π/4 and π/4)
            demo_rotation = np.random.uniform(-np.pi/4, np.pi/4)
            
            # Update the existing environment with new rotation
            env.boundary_angle = demo_rotation
            
            # Set target drawing based on previous episode (if available)
            if episode_count == 1:
                # Use the previous episode's drawing as the target for this episode
                env.set_target_drawing(first_demo_drawing_image, first_demo_boundary_angle)
                if verbose:
                    print(f"    Using previous episode's drawing as target (episode {episode_count})")
            elif episode_count == 0:
                # First episode has no target (None for new tasks)
                env.set_target_drawing(None)
                if verbose:
                    print(f"    First episode - no target drawing set")
            
            # Sample speeds for this demo (different from other demos in the same task)
            trajectory_speed = np.random.uniform(trajectory_speed_min, trajectory_speed_max)
            
            # Convert control points to actions at the sampled speed for this demo
            trajectory_actions_upright, total_trajectory_steps = convert_control_points_to_actions(
                trajectory_control_points, control_hz, trajectory_speed, noise_std, noise_bounds, board_length, margin,
                offset_magnitude_factor, offset_magnitude_max,
                part_delay_min, part_delay_max, min_distance)
            
            # Apply rotation transformation to the generated trajectory for this demo
            center_x, center_y = 256.0, 256.0
            rotated_actions = []
            
            for action in trajectory_actions_upright:
                x, y, pen_down = action
                
                # Translate to origin
                x_rel = x - center_x
                y_rel = y - center_y
                
                # Apply rotation
                cos_rot = np.cos(demo_rotation)
                sin_rot = np.sin(demo_rotation)
                x_rotated = x_rel * cos_rot - y_rel * sin_rot
                y_rotated = x_rel * sin_rot + y_rel * cos_rot
                
                # Translate back
                x_final = x_rotated + center_x
                y_final = y_rotated + center_y
                
                rotated_actions.append(np.array([x_final, y_final, pen_down], dtype=np.float32))
            
            trajectory_actions = np.array(rotated_actions)
            start_pos_rotated = np.array([trajectory_actions[0][0], trajectory_actions[0][1]])
            
            # reset env and get observations
            env.seed(process_seed)  # Use deterministic seed based on task and demo
            obs = env.reset(no_rotation=False)
            img = (np.transpose(obs['image'], (1,2,0)) * 255).astype(np.uint8)
            
            def execute_step(act, step_name=""):
                """Helper function to execute a step and record data"""
                nonlocal obs, img
                
                # Record step data
                img_data = img.astype(np.uint8)
                agent_pos_data = np.array(obs['agent_pos'], dtype=np.float32)
                pen_down_data = np.array(obs['pen_down'], dtype=np.float32)
                action_data = act
                
                # Store data for this step
                episode_data['image'].append(img_data)
                episode_data['agent_pos'].append(agent_pos_data)
                episode_data['pen_down'].append(pen_down_data)
                episode_data['action'].append(action_data)
                
                # store labels for this step
                episode_labels['boundary_angle'].append(np.array(demo_rotation, dtype=np.float32))
                episode_labels['drawing_image'].append(env.get_drawing_image())
                
                # step env and render
                obs, reward, done, info = env.step(act)
                img = (np.transpose(obs['image'], (1,2,0)) * 255).astype(np.uint8)
                
                # Regulate control frequency only if viewer is enabled
                if viewer:
                    pygame.time.wait(int(1000 / control_hz))
            
            # Stage 1: Move cursor to starting position (pen up) using Bezier curve
            if verbose:
                print(f"    Moving cursor to starting position...")
            
            # Sample random number of positioning steps for this episode
            num_positioning_steps = np.random.randint(positioning_steps_min, positioning_steps_max + 1)
            
            cursor_at_start = False
            
            # Generate Bezier curve control points for smooth positioning movement (once)
            # Create control points that ensure smooth path from current to target
            current_pos = np.array(obs['agent_pos'], dtype=np.float32)
            
            # Use shared helper function for smooth movement control points
            control1, control2 = generate_movement_control_points(current_pos, start_pos_rotated, board_length, margin,
                                                               offset_magnitude_factor, offset_magnitude_max)
            
            for positioning_step in range(num_positioning_steps):
                # Get current cursor position
                current_pos = np.array(obs['agent_pos'], dtype=np.float32)
                
                # Check if cursor is close enough to start position
                distance_to_start = np.linalg.norm(current_pos - start_pos_rotated)
                if verbose:
                    print(f"    Distance to start: {distance_to_start:.1f}")
                if distance_to_start < 5.0:  # Within 5 pixels
                    cursor_at_start = True
                    if verbose:
                        print(f"    Cursor positioned at start (distance: {distance_to_start:.1f})")
                    break
                
                # Calculate smooth progression along the Bezier curve
                t = positioning_step / (num_positioning_steps - 1) if num_positioning_steps > 1 else 0
                
                # Use helper function for Bezier curve computation
                target_pos = compute_bezier_point(current_pos, control1, control2, start_pos_rotated, t)
                
                # Move cursor towards target position (pen up)
                act = np.array([target_pos[0], target_pos[1], 0.0], dtype=np.float32)
                execute_step(act, "positioning")

            # wait for cursor to reach start position
            timeout_steps = 100
            while not cursor_at_start:
                current_pos = np.array(obs['agent_pos'], dtype=np.float32)
                distance_to_start = np.linalg.norm(current_pos - start_pos_rotated)
                if verbose:
                    print(f"    Distance to start: {distance_to_start:.1f}")
                    print(f"    Cursor not at start (distance: {distance_to_start:.1f})")
                if distance_to_start < 5.0:
                    cursor_at_start = True
                    if verbose:
                        print(f"    Cursor positioned at start (distance: {distance_to_start:.1f})")
                    break
                act = np.array([start_pos_rotated[0], start_pos_rotated[1], 0.0], dtype=np.float32)
                execute_step(act, "positioning")
                timeout_steps -= 1
                if timeout_steps <= 0:
                    raise ValueError(f"Cursor positioning took too long")
            
            # Stage 2: Execute procedural trajectory (pen down)
            if verbose:
                print(f"    Executing procedural trajectory at {trajectory_speed:.1f} px/s...")
            
            for step_idx in range(total_trajectory_steps):
                # Get the pre-generated action for this step
                act = trajectory_actions[step_idx]
                execute_step(act, "trajectory")
            
            # Validate reward for episodes after the first (if they have a target drawing)
            if episode_count > 0 and first_demo_drawing_image is not None:
                # Get the final reward from the environment
                final_reward = env.compute_reward()
                if verbose:
                    print(f"    Final reward: {final_reward:.3f} (threshold: {reward_threshold})")
                
                # Check if reward meets the threshold
                if final_reward < reward_threshold:
                    error_msg = f"Episode {episode_count} reward {final_reward:.3f} below threshold {reward_threshold} for task '{task_name}'"
                    print(f"    REWARD THRESHOLD NOT MET: {error_msg}")
                    print(f"    Task {task_idx}: Retrying demo {demo_idx+1} with different seed...")
                    # Increment retry count and continue to next iteration
                    retry_count += 1
                    continue
                
                if verbose:
                    print(f"    Reward validation passed: {final_reward:.3f} >= {reward_threshold}")
            
            # If we reach here, the demo was successful
            demo_success = True
            
            # Convert lists to numpy arrays
            for key in episode_data:
                episode_data[key] = np.stack(episode_data[key])
            
            # Create a single task that spans the entire episode
            total_episode_steps = len(episode_data['image'])
            task_data = [{
                "name": task_name,
                "start_idx": 0, 
                "end_idx": total_episode_steps,
                "labels": episode_labels
            }]
            
            # Add episode to replay buffer
            current_replay_buffer.add_episode(
                data=episode_data,
                tasks=task_data,
                episode_name=f"episode_{episode_count:04d}"
            )
            
            if verbose:
                print(f'  Generated demo {demo_idx+1}/{demos_per_task} with rotation {demo_rotation:.3f} rad at {trajectory_speed:.1f} px/s')
            
            # Store this episode's drawing image and boundary angle for the next episode
            if episode_count == 0:
                first_demo_drawing_image = env.get_drawing_image()
                first_demo_boundary_angle = demo_rotation
            
            episode_count += 1
        
        # Check if we failed to generate this demo after all retries
        if not demo_success:
            error_msg = f"Failed to generate demo {demo_idx+1} for task '{task_name}' after {max_retries} retries"
            print(f"    ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    if verbose:
        print(f'Completed task "{task_name}" with {demos_per_task} demos')
    
    # Save replay buffer to disk before video generation
    output_path = get_task_dataset_path(output, task_name)
    current_replay_buffer.save_to_path(output_path)
    if verbose:
        print(f"    Saved replay buffer to disk")
    
    # Generate video for this task if visualize flag is set
    if visualize:
        if verbose:
            print(f"    Generating video for task '{task_name}'...")
        
        # Generate video for this task using the replay buffer object directly
        print_replay_buffer_draw(output_path, replay_buffer=current_replay_buffer, print_summary=False, vis_video=True, enable_print=verbose)
        if verbose:
            print(f"    Video generated successfully for '{task_name}'")
    
    return (task_name, episode_count)

def generate_procedural_demos(output, control_hz=10, boundary_angle=None, persistent=True, 
                            num_tasks=10, demos_per_task=5, 
                            trajectory_speed_min=100.0, trajectory_speed_max=300.0,
                            margin=20, noise_std=1.0, noise_bounds=20.0, min_parts=3, max_parts=5, 
                            overwrite=False, visualize=False, viewer=False, max_workers=None, base_seed=0,
                            positioning_steps_min=20, positioning_steps_max=60,
                            offset_magnitude_factor=0.3, offset_magnitude_max=100.0, reward_threshold=-5.0, connection_probability=0.5,
                            part_delay_min=0, part_delay_max=10, min_distance=50.0,
                            oval_major_axis_min=30.0, oval_major_axis_max=80.0, partial_oval_probability=0.0, allowed_parts=None):
    """
    Generate procedural drawing demonstrations using multiprocessing.
    
    Args:
        output: Output directory for the replay buffer
        control_hz: Control frequency in Hz
        boundary_angle: Boundary rotation angle in radians (unused in multiprocessing version)
        persistent: Whether to keep the replay buffer after generation
        num_tasks: Number of tasks to generate
        demos_per_task: Number of demonstrations per task
        trajectory_speed_min: Minimum trajectory speed in pixels per second
        trajectory_speed_max: Maximum trajectory speed in pixels per second
        margin: Margin from board edge in pixels
        noise_std: Standard deviation of Gaussian noise to add to trajectory positions
        noise_bounds: Maximum absolute value for noise clipping (in pixels). Set to None to disable clipping.
        min_parts: Minimum number of parts in each trajectory
        max_parts: Maximum number of parts in each trajectory
        overwrite: Whether to overwrite existing output directory
        visualize: Whether to generate videos for all datasets after generation
        viewer: Whether to enable the visual viewer window
        max_workers: Maximum number of worker processes (None = auto-detect)
        base_seed: Base seed to offset all random seeds used in generation
        positioning_steps_min: Minimum number of steps for positioning stage
        positioning_steps_max: Maximum number of steps for positioning stage
        offset_magnitude_factor: Factor to multiply distance by for movement control point offsets
        offset_magnitude_max: Maximum offset magnitude in pixels for movement control points
        reward_threshold: Minimum acceptable reward threshold for drawing similarity
        connection_probability: Probability that a new part connects to a previous part endpoint
        part_delay_min: Minimum number of delay steps between trajectory parts (default: 0)
        part_delay_max: Maximum number of delay steps between trajectory parts (default: 10)
        min_distance: Minimum distance between start and end positions for each trajectory part in pixels (default: 50.0)
        oval_major_axis_min: Minimum major axis size for oval parts in pixels (default: 30.0)
        oval_major_axis_max: Maximum major axis size for oval parts in pixels (default: 80.0)
        partial_oval_probability: Probability of drawing partial ovals instead of full ovals (default: 0.0)
        allowed_parts: List of allowed part types ['straight', 'curve', 'oval', 'movement'] or None for all (default: None)
    """
    # Check if output directory already exists
    resume_mode = False
    tasks_to_generate = list(range(num_tasks))  # Default: generate all tasks
    
    if os.path.exists(output):
        if overwrite:
            print(f"Output directory '{output}' already exists. Removing it due to --overwrite flag...")
            shutil.rmtree(output)
        else:
            # Resume mode: check for existing completed tasks
            print(f"Output directory '{output}' already exists. Checking for completed tasks...")
            completed_tasks, missing_tasks, cleaned_tasks = get_completed_tasks(output, num_tasks, demos_per_task, check_videos=visualize, max_workers=max_workers)
            
            if len(completed_tasks) > 0 or len(cleaned_tasks) > 0:
                resume_mode = True
                tasks_to_generate = missing_tasks
                print(f"Found {len(completed_tasks)} completed tasks, {len(cleaned_tasks)} partial tasks cleaned, {len(missing_tasks)} tasks remaining to generate")
                if len(missing_tasks) == 0:
                    print("All tasks are already completed! Nothing to do.")
                    return
            else:
                print(f"No completed tasks found in existing directory.")
    
    # create output directory
    os.makedirs(output, exist_ok=True)
    cleanup_empty_tasks(output)
    
    def print_task_summary(new_task_name=None):
        """Print a summary of all tasks and their demo counts."""
        msg = f"======== Task Summary for {output} ========"
        print(f'\n{msg}')
        all_task_names = get_all_task_names(output, new_task_name)
        if all_task_names:
            total_demos = 0
            for cur_task_name in all_task_names:
                replay_buffer = get_or_create_replay_buffer_for_task(output, cur_task_name, in_memory=False, ignore_empty=True)
                demo_count = replay_buffer.n_episodes
                total_demos += demo_count
                print(f'  "{cur_task_name}": {demo_count} demos')
            print(f'\nTotal: {len(all_task_names)} tasks, {total_demos} demos')
        else:
            print('  No tasks found')
        print('=' * len(msg) + '\n')
    
    # Get board length from a temporary environment
    temp_env = DrawEnv(boundary_angle=0, render_mode='rgb_array')
    board_length = temp_env.board_length
    del temp_env  # Clean up temporary environment
    
    if resume_mode:
        print(f"RESUME MODE: Generating {len(tasks_to_generate)} remaining tasks with {demos_per_task} demos each...")
        print(f"Total episodes to generate: {len(tasks_to_generate) * demos_per_task}")
    else:
        print(f"Generating {num_tasks} tasks with {demos_per_task} demos each...")
        print(f"Total episodes to generate: {num_tasks * demos_per_task}")
    print(f"Positioning steps range: {positioning_steps_min}-{positioning_steps_max} steps")
    print(f"Trajectory speed range: {trajectory_speed_min}-{trajectory_speed_max} px/s")
    print(f"Part delay range: {part_delay_min}-{part_delay_max} steps")
    print(f"Minimum part distance: {min_distance} pixels")
    
    # Determine number of worker processes
    if max_workers is None:
        max_workers = min(len(tasks_to_generate), multiprocessing.cpu_count())
    
    # Check if we're running single-process
    is_single_process = (max_workers == 1)
    
    if is_single_process:
        print(f"Running in single-process mode")
    else:
        print(f"Using {max_workers} worker processes")
    
    start_time = time.time()
    
    if is_single_process:
        # Single-process execution - simpler and more verbose
        total_episodes = 0
        successful_tasks = 0
        
        for task_idx in tasks_to_generate:
            args = (output, task_idx, num_tasks, control_hz, board_length, margin, demos_per_task,
                   trajectory_speed_min, trajectory_speed_max,
                   noise_std, noise_bounds, min_parts, max_parts, visualize, viewer, True, base_seed,
                   positioning_steps_min, positioning_steps_max, offset_magnitude_factor, offset_magnitude_max, reward_threshold, connection_probability,
                   part_delay_min, part_delay_max, min_distance, oval_major_axis_min, oval_major_axis_max, partial_oval_probability, allowed_parts)  # verbose=True for single process
            task_name, episode_count = generate_single_task(args)
            successful_tasks += 1
            total_episodes += episode_count
        
    else:
        # Multi-process execution with progress tracking
        total_episodes = 0
        successful_tasks = 0
        
        # Prepare arguments for missing tasks only
        task_args = []
        for task_idx in tasks_to_generate:
            args = (output, task_idx, num_tasks, control_hz, board_length, margin, demos_per_task,
                   trajectory_speed_min, trajectory_speed_max,
                   noise_std, noise_bounds, min_parts, max_parts, visualize, viewer, False, base_seed,
                   positioning_steps_min, positioning_steps_max, offset_magnitude_factor, offset_magnitude_max, reward_threshold, connection_probability,
                   part_delay_min, part_delay_max, min_distance, oval_major_axis_min, oval_major_axis_max, partial_oval_probability, allowed_parts)  # verbose=False for multiprocessing
            task_args.append(args)
        
        # Use multiprocessing Pool for parallel task generation
        with Pool(processes=max_workers) as pool:
            # Submit missing tasks to the process pool and track progress with tqdm
            results = []
            with tqdm(total=len(tasks_to_generate), desc="Generating tasks", unit="task") as pbar:
                for result in pool.imap_unordered(generate_single_task, task_args):
                    results.append(result)
                    pbar.update(1)
            
            # Process results
            for result in results:
                task_name, episode_count = result
                successful_tasks += 1
                total_episodes += episode_count
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    if resume_mode:
        print(f"Successful tasks (this run): {successful_tasks}/{len(tasks_to_generate)}")
        print(f"Total tasks completed: {num_tasks - len(tasks_to_generate) + successful_tasks}/{num_tasks}")
    else:
        print(f"Successful tasks: {successful_tasks}/{num_tasks}")
    print(f"Total episodes generated (this run): {total_episodes}")
    print(f"Total time: {total_time:.2f} seconds")
    if len(tasks_to_generate) > 0:
        print(f"Average time per task: {total_time/len(tasks_to_generate):.2f} seconds")
    print(f"Average time per episode: {total_time/total_episodes:.2f} seconds" if total_episodes > 0 else "No episodes generated")
    
    # Print final task summary
    print_task_summary()
        
    # At the end of the script, delete the replay buffer if persistent is False
    if not persistent:
        try:
            if input(f'Press Y to delete the replay buffer directory {output}: ') == 'Y':
                shutil.rmtree(output)
                print(f"Replay buffer directory '{output}' deleted as persistent=False.")
        except Exception as e:
            print(f"Failed to delete replay buffer directory '{output}': {e}")

def check_and_cleanup_single_task(args):
    """
    Check if a single task is completed and clean it up if needed. This function is designed to run in a separate process.
    
    Args:
        args: Tuple containing (output, task_idx, num_tasks, demos_per_task, check_videos)
    
    Returns:
        tuple: (task_idx, is_completed, was_cleaned, cleanup_messages)
    """
    output, task_idx, num_tasks, demos_per_task, check_videos = args
    
    # Calculate the number of digits needed based on total tasks
    num_digits = len(str(num_tasks))
    task_number = str(task_idx + 1).zfill(num_digits)
    task_name = f'draw procedural_{task_number}'
    
    cleanup_messages = []
    was_cleaned = False
    
    try:
        # Check if the replay buffer exists and has the expected number of demos
        replay_buffer = get_or_create_replay_buffer_for_task(output, task_name, in_memory=False, ignore_empty=True)
        
        task_complete = replay_buffer.n_episodes >= demos_per_task
        
        # If videos are expected, also check for video file existence
        if task_complete and check_videos:
            # Convert spaces to underscores for video filename
            video_task_name = task_name.replace(' ', '_')
            video_filename = f"tmp_{video_task_name}_lower_draw_video.mp4"
            video_path = os.path.join(output, video_filename)
            if not os.path.exists(video_path):
                task_complete = False
        
        if task_complete:
            return (task_idx, True, False, [])
        else:
            # Partial task detected - clean it up
            task_path = get_task_dataset_path(output, task_name)
            if os.path.exists(task_path):
                if replay_buffer.n_episodes < demos_per_task:
                    cleanup_messages.append(f"Cleaning partial task '{task_name}' ({replay_buffer.n_episodes}/{demos_per_task} demos)")
                else:
                    cleanup_messages.append(f"Cleaning task '{task_name}' (missing video file)")
                shutil.rmtree(task_path)
                was_cleaned = True
                
                # Also clean up any orphaned video file
                video_task_name = task_name.replace(' ', '_')
                video_filename = f"tmp_{video_task_name}_lower_draw_video.mp4"
                video_path = os.path.join(output, video_filename)
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        cleanup_messages.append(f"Removed orphaned video file: {video_filename}")
                    except Exception as e:
                        cleanup_messages.append(f"Warning: Failed to remove orphaned video file {video_filename}: {e}")
            
            return (task_idx, False, was_cleaned, cleanup_messages)
            
    except Exception:
        # If there's any error loading the replay buffer, try to clean up any existing files
        task_path = get_task_dataset_path(output, task_name)
        if os.path.exists(task_path):
            cleanup_messages.append(f"Cleaning corrupted task '{task_name}' (load error)")
            try:
                shutil.rmtree(task_path)
                was_cleaned = True
            except Exception as e:
                cleanup_messages.append(f"Warning: Failed to clean corrupted task '{task_name}': {e}")
        
        # Also clean up any orphaned video file
        if check_videos:
            video_task_name = task_name.replace(' ', '_')
            video_filename = f"tmp_{video_task_name}_lower_draw_video.mp4"
            video_path = os.path.join(output, video_filename)
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    cleanup_messages.append(f"Removed orphaned video file: {video_filename}")
                except Exception as e:
                    cleanup_messages.append(f"Warning: Failed to remove orphaned video file {video_filename}: {e}")
        
        return (task_idx, False, was_cleaned, cleanup_messages)

def get_completed_tasks(output, num_tasks, demos_per_task, check_videos=False, max_workers=None):
    """
    Check which tasks are already completed in the output directory using multiprocessing.
    Deletes partial/incomplete replay buffers to ensure clean regeneration.
    
    Args:
        output: Output directory path
        num_tasks: Total number of tasks expected
        demos_per_task: Expected number of demos per task
        check_videos: Whether to also check for video files (when visualize=True)
        max_workers: Maximum number of worker processes (None = auto-detect)
    
    Returns:
        completed_tasks: Set of task indices (0-based) that are already completed
        missing_tasks: List of task indices (0-based) that need to be generated
        cleaned_tasks: List of task indices (0-based) that had partial data and were cleaned up
    """
    completed_tasks = set()
    missing_tasks = []
    cleaned_tasks = []
    
    # Determine number of worker processes
    if max_workers is None:
        max_workers = min(num_tasks, multiprocessing.cpu_count())
    
    # Check if we're running single-process
    is_single_process = (max_workers == 1)
    
    # Prepare arguments for all tasks
    task_args = []
    for task_idx in range(num_tasks):
        args = (output, task_idx, num_tasks, demos_per_task, check_videos)
        task_args.append(args)
    
    if is_single_process:
        # Single-process execution
        for args in tqdm(task_args, desc="Checking tasks", total=num_tasks):
            task_idx, is_completed, was_cleaned, cleanup_messages = check_and_cleanup_single_task(args)
            
            if is_completed:
                completed_tasks.add(task_idx)
            else:
                missing_tasks.append(task_idx)
                if was_cleaned:
                    cleaned_tasks.append(task_idx)
            
            # Print cleanup messages
            for message in cleanup_messages:
                print(f"  {message}")
    else:
        # Multi-process execution
        with Pool(processes=max_workers) as pool:
            results = []
            with tqdm(total=num_tasks, desc="Checking tasks", unit="task") as pbar:
                for result in pool.imap_unordered(check_and_cleanup_single_task, task_args):
                    results.append(result)
                    pbar.update(1)
            
            # Process results
            for task_idx, is_completed, was_cleaned, cleanup_messages in results:
                if is_completed:
                    completed_tasks.add(task_idx)
                else:
                    missing_tasks.append(task_idx)
                    if was_cleaned:
                        cleaned_tasks.append(task_idx)
                
                # Print cleanup messages
                for message in cleanup_messages:
                    print(f"  {message}")
    
    return completed_tasks, missing_tasks, cleaned_tasks

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-hz', '--control_hz', default=10, type=int)
@click.option('--boundary-angle', type=float, default=None, help='Boundary rotation angle in radians. If not specified, will be random within the range -π/4 to π/4.')
@click.option('--persistent/--no-persistent', default=True, help='If set to --no-persistent, the replay buffer file will be deleted at the end of the script.')
@click.option('--num-tasks', default=10, type=int, help='Number of tasks to generate.')
@click.option('--demos-per-task', default=5, type=int, help='Number of demonstrations per task.')
@click.option('--trajectory-speed-min', default=100.0, type=float, help='Minimum trajectory speed in pixels per second.')
@click.option('--trajectory-speed-max', default=300.0, type=float, help='Maximum trajectory speed in pixels per second.')
@click.option('--margin', default=0, type=int, help='Margin from board edge in pixels.')
@click.option('--noise-std', default=3.0, type=float, help='Standard deviation of Gaussian noise to add to trajectory positions (in pixels).')
@click.option('--noise-bounds', default=20.0, type=float, help='Maximum absolute value for noise clipping (in pixels). Set to None to disable clipping.')
@click.option('--min-parts', default=1, type=int, help='Minimum number of parts in each trajectory.')
@click.option('--max-parts', default=6, type=int, help='Maximum number of parts in each trajectory.')
@click.option('--overwrite', is_flag=True, help='Overwrite existing output directory if it exists.')
@click.option('--visualize/--no-visualize', default=True, help='Generate videos for all datasets after generation (default: True).')
@click.option('--viewer/--no-viewer', default=False, help='Enable/disable the visual viewer window.')
@click.option('--max-workers', default=None, type=int, help='Maximum number of worker processes (None = auto-detect).')
@click.option('--base-seed', default=0, type=int, help='Base seed to offset all random seeds used in generation.')
@click.option('--positioning-steps-min', default=5, type=int, help='Minimum number of steps for positioning stage.')
@click.option('--positioning-steps-max', default=20, type=int, help='Maximum number of steps for positioning stage.')
@click.option('--offset-magnitude-factor', default=0.3, type=float, help='Factor to multiply distance by for movement control point offsets (default: 0.3).')
@click.option('--offset-magnitude-max', default=100.0, type=float, help='Maximum offset magnitude in pixels for movement control points (default: 100.0).')
@click.option('--reward-threshold', default=-10.0, type=float, help='Minimum acceptable reward threshold for drawing similarity (default: -5.0).')
@click.option('--connection-probability', default=0.5, type=float, help='Probability that a new part connects to a previous part endpoint (default: 0.5).')
@click.option('--part-delay-min', default=0, type=int, help='Minimum number of delay steps between trajectory parts (default: 0).')
@click.option('--part-delay-max', default=10, type=int, help='Maximum number of delay steps between trajectory parts (default: 10).')
@click.option('--min-distance', default=50, type=float, help='Minimum distance between start and end positions for each trajectory part in pixels (default: 50).')
@click.option('--oval-major-axis-min', default=30, type=float, help='Minimum major axis size for oval parts in pixels (default: 30).')
@click.option('--oval-major-axis-max', default=200, type=float, help='Maximum major axis size for oval parts in pixels (default: 80).')
@click.option('--partial-oval-probability', default=0.5, type=float, help='Probability of drawing partial ovals instead of full ovals (default: 0.0).')
@click.option('--parts', default=None, help='Comma-separated list of allowed part types (straight,curve,oval,movement). If not specified, all types are allowed.')
@click.option('--group-demos/--no-group-demos', 'group_demos_flag', default=True, help='Group all generated demos into a single zarr.zip file after generation (default: True).')
@click.option('--vis-grouped-demos/--no-vis-grouped-demos', 'vis_grouped_demos', default=True, help='Create visualization grids (image and video) from grouped demos (default: True). Only used if group_demos_flag is True.')
@click.option('--vis-max-videos', default=100, type=int, help='Maximum number of videos to use for visualization grids (default: 100).')
@click.option('--vis-max-duration', default=60, type=float, help='Maximum duration in seconds for video grid (default: 60).')
def main_click(output, control_hz, boundary_angle, persistent, num_tasks, demos_per_task, 
               trajectory_speed_min, trajectory_speed_max, 
               margin, noise_std, noise_bounds, min_parts, max_parts, overwrite, visualize, viewer, max_workers, base_seed,
               positioning_steps_min, positioning_steps_max, offset_magnitude_factor, offset_magnitude_max, reward_threshold,
               connection_probability, part_delay_min, part_delay_max, min_distance,
               oval_major_axis_min, oval_major_axis_max, partial_oval_probability, parts, group_demos_flag, vis_grouped_demos,
               vis_max_videos, vis_max_duration):
    """
    Procedurally generate drawing data for the Draw task using behavior_prompting replay buffer format.
    
    Usage: python procedural_generate_drawings.py -o datasets/draw/procedural --num-tasks 20 --demos-per-task 5
    
    This script generates drawing data automatically without user input.
    It creates multiple tasks, each with multiple demonstrations at different whiteboard rotations.
    Each task has the same procedural drawing trajectory but rotated to match the board rotation.
    Movement speeds are randomly sampled from specified ranges for each demo.
    Small amounts of noise are added to trajectory positions for realism and variety.
    Trajectory complexity (number of parts) is configurable via --min-parts and --max-parts.
    
    QUALITY CONTROL: After the first episode in each task, subsequent episodes use the previous
    episode's drawing as a target. The environment computes a reward based on drawing similarity,
    and episodes must meet a minimum reward threshold (--reward-threshold). If the reward threshold
    is not met, the script will automatically retry the demo generation with different random seeds
    up to 10 times before failing.
    
    DRAWING CONNECTIONS: With --connection-probability, new trajectory parts can connect to
    previous part endpoints, creating more realistic and connected drawings instead of completely
    random positions.
    
    MULTIPROCESSING: This script now uses multiprocessing for parallel task generation.
    Each process works on a separate task, generating all episodes for that task independently.
    Use --max-workers to control the number of parallel processes.
    
    SEEDING: Use --base-seed to set a base seed that offsets all random seeds used in generation.
    This ensures reproducible results across different runs while maintaining variety between tasks.
    
    POSITIONING: The positioning stage uses a step-based approach where each episode samples
    a random number of steps between --positioning-steps-min and --positioning-steps-max.
    This creates smooth Bezier curve paths to the starting position.
    
    Task Naming Convention:
    - Task names are automatically generated with leading zeros based on total number of tasks
    - Examples: For 10 tasks: "draw procedural01", "draw procedural02", ..., "draw procedural10"
    - Examples: For 100 tasks: "draw procedural001", "draw procedural002", ..., "draw procedural100"  
    - Examples: For 1000 tasks: "draw procedural0001", "draw procedural0002", ..., "draw procedural1000"
    - Each task has multiple demos with the same drawing but different rotations
    
    Example folder structure (for 10 tasks):
    datasets/draw/procedural/
    ├── draw_procedural01.zarr/
    ├── draw_procedural02.zarr/
    └── draw_procedural10.zarr/
    
    The script will generate num_tasks * demos_per_task total episodes.
    
    Output Options:
    - Use --overwrite to remove existing output directory and start fresh
    - If output directory exists WITHOUT --overwrite, the script will automatically resume:
      * Checks which tasks are already completed (have the expected number of demos)
      * Automatically cleans up partial/incomplete tasks for safe regeneration
      * Only generates the missing/incomplete tasks
      * Preserves all existing completed tasks
    - This allows recovery from interrupted runs without losing progress or data corruption
    
    Visualization Options:
    - Use --visualize to generate videos for all datasets after generation
    - Use --viewer to enable the visual window during generation
    - Use --no-viewer to disable the visual window for faster generation (default)
    
    Multiprocessing Options:
    - Use --max-workers to specify the number of parallel processes
    - Default is auto-detection based on CPU cores and number of tasks
    
    Seeding Options:
    - Use --base-seed to set a base seed for reproducible generation
    - Different base seeds will produce different trajectories while maintaining consistency within each run
    
    Positioning Options:
    - Use --positioning-steps-min and --positioning-steps-max to control positioning duration
    - Each episode samples a random number of steps within this range
    - More steps = smoother, slower positioning movement
    
    Movement Control Options:
    - Use --offset-magnitude-factor to control how much perpendicular offset is applied to movement control points
      (higher values create more curved movement paths)
    - Use --offset-magnitude-max to set the maximum offset in pixels
      (prevents extremely large offsets for very long movements)
    
    Drawing Connection Options:
    - Use --connection-probability to control how often new parts connect to previous part endpoints
      (higher values create more connected/overlapping drawings)
    
    Part Delay Options:
    - Use --part-delay-min and --part-delay-max to control the delay between trajectory parts
      (delays consist of repeating the end position action, giving time for the cursor to reach target positions)
    
    Distance Control Options:
    - Use --min-distance to set the minimum distance between start and end positions for each trajectory part
      (prevents very short segments and ensures meaningful drawing movements)
    
    Quality Control Options:
    - Use --reward-threshold to set the minimum acceptable reward for drawing similarity
      (episodes after the first will use the previous episode's drawing as a target and validate the reward)
      (if reward threshold is not met, the script will automatically retry up to 10 times with different seeds)
    
    Partial Oval Options:
    - Use --partial-oval-probability to set the probability of drawing partial ovals instead of full ovals
      (0.0 = always full ovals, 1.0 = always partial ovals, 0.5 = 50% chance of partial ovals)
      (partial ovals have random angular ranges between π/8 and 2π radians)
    
    Part Type Filtering (for debugging):
    - Use --parts to restrict which part types can be generated
      (e.g., --parts oval generates only ovals, --parts straight,curve generates only straight lines and curves)
      (valid types: straight, curve, oval, movement)
      (if not specified, all types are allowed)
    
    Demo Grouping Options:
    - Use --group-demos to automatically group all generated demos into a single zarr.zip file after generation (default: True)
    - Use --no-group-demos to skip the grouping step and keep individual zarr directories
      (the grouped file will be saved as {output_directory}.zarr.zip)
    
    Visualization Options (for grouped demos):
    - Use --vis-grouped-demos to create visualization grids (image and video) from grouped demos (default: True)
      (only used when --group-demos is enabled)
    - Use --no-vis-grouped-demos to skip visualization grid creation
    - Use --vis-max-videos to set maximum number of videos for visualization grids (default: 100)
    - Use --vis-max-duration to set maximum duration in seconds for video grid (default: 150)
    """
    # Convert output path to absolute
    output = os.path.abspath(output)
    
    # Parse allowed parts
    allowed_parts = None
    if parts is not None:
        allowed_parts = [part.strip() for part in parts.split(',')]
        valid_parts = ['straight', 'curve', 'oval', 'movement']
        for part in allowed_parts:
            if part not in valid_parts:
                raise click.ClickException(f"Invalid part type '{part}'. Valid types are: {valid_parts}")
        print(f"Restricting to part types: {allowed_parts}")
    else:
        print("Using all part types: straight, curve, oval, movement")

    start_time = time.time()
    
    generate_procedural_demos(output, control_hz=control_hz, boundary_angle=boundary_angle, 
                            persistent=persistent, num_tasks=num_tasks, 
                            demos_per_task=demos_per_task, 
                            trajectory_speed_min=trajectory_speed_min,
                            trajectory_speed_max=trajectory_speed_max,
                            margin=margin, noise_std=noise_std, noise_bounds=noise_bounds, min_parts=min_parts, max_parts=max_parts, 
                            overwrite=overwrite, visualize=visualize, viewer=viewer, max_workers=max_workers,
                            base_seed=base_seed, positioning_steps_min=positioning_steps_min, 
                            positioning_steps_max=positioning_steps_max,
                            offset_magnitude_factor=offset_magnitude_factor,
                            offset_magnitude_max=offset_magnitude_max, reward_threshold=reward_threshold,
                            connection_probability=connection_probability,
                            part_delay_min=part_delay_min, part_delay_max=part_delay_max,
                            min_distance=min_distance,
                            oval_major_axis_min=oval_major_axis_min, oval_major_axis_max=oval_major_axis_max,
                            partial_oval_probability=partial_oval_probability, allowed_parts=allowed_parts)
    
    # Optionally group all demos into a single zarr.zip file
    if group_demos_flag:
        print(f"\nGrouping all demos into a single zarr.zip file...")
        output_zip_path = output + ".zarr.zip"
        group_demos(
            input_dirs=[output], 
            output=output_zip_path, 
            num_workers=max_workers,
            create_grids=vis_grouped_demos,
            vis_max_videos=vis_max_videos,
            vis_max_duration=vis_max_duration
        )
        print(f"Successfully grouped demos into: {output_zip_path}")

    end_time = time.time()
    print(f"Total time: {(end_time - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main_click()
