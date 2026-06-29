"""
Generates BDDL files given a yaml specification of the new tasks to generate. The idea is to generate new tasks by taking existing LIBERO environments and taking combinations of existing objects to do new actions with.

This version uses an object-centric approach where objects define their supported operators,
and the script automatically generates all valid task combinations.

CHAINING SUPPORT:
This script supports generating multi-action chained tasks. Objects can define chaining
properties in the YAML to control what actions are valid after other actions:
- can_grasp_with_contents: Whether an object can be picked up when something is on/in it
- max_items_on_top: How many items can be placed on/in an object

Operators can define ordering constraints:
- requires_action_replay: Must be the first action (for open, turn_on)
Note: Actions requiring replay must be first regardless of other properties.
Some actions like turn_on don't affect other operators' validity but still require replay.

Parts adapted from LIBERO Pro
"""

import argparse
import time
import yaml
import os
import re
import shutil
import copy
import numpy as np
import random
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from behavior_prompting.train_network import fix_robosuite_log_permission_issue
fix_robosuite_log_permission_issue()
from generate_init_states import generate_init_states_for_single_bddl

from libero.libero import get_libero_path


class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that disables anchors/aliases for repeated structures."""

    def ignore_aliases(self, data):
        return True


@dataclass
class WorldState:
    """Tracks the state of objects during chained task generation."""
    # Map from object -> list of objects on/in it
    objects_on_top: Dict[str, List[str]] = field(default_factory=dict)
    # Set of objects that have been used (grasped and placed somewhere)
    placed_objects: Set[str] = field(default_factory=set)
    # Set of objects that have been opened (drawers)
    opened_objects: Set[str] = field(default_factory=set)
    # Set of objects that have been turned on
    turned_on_objects: Set[str] = field(default_factory=set)
    # Whether an action requiring replay has been used (only one allowed, must be first)
    action_replay_used: bool = False
    # The first action (used to track ordering constraints)
    first_action_type: Optional[str] = None
    
    def copy(self) -> 'WorldState':
        """Create a deep copy of the world state."""
        new_state = WorldState()
        new_state.objects_on_top = {k: list(v) for k, v in self.objects_on_top.items()}
        new_state.placed_objects = set(self.placed_objects)
        new_state.opened_objects = set(self.opened_objects)
        new_state.turned_on_objects = set(self.turned_on_objects)
        new_state.action_replay_used = self.action_replay_used
        new_state.first_action_type = self.first_action_type
        return new_state
    
    def get_items_on_object(self, obj_name: str) -> List[str]:
        """Get items currently on/in an object."""
        return self.objects_on_top.get(obj_name, [])
    
    def count_items_on_object(self, obj_name: str) -> int:
        """Count how many items are on/in an object."""
        return len(self.get_items_on_object(obj_name))
    
    def has_contents(self, obj_name: str) -> bool:
        """Check if an object has anything on/in it."""
        return self.count_items_on_object(obj_name) > 0
    
    def is_object_available_for_grasp(self, obj_name: str, objects_config: Dict) -> bool:
        """Check if an object can be grasped given current state."""
        # Can't grasp if already placed somewhere
        if obj_name in self.placed_objects:
            return False
        
        # Check if object has contents and whether it can be grasped with contents
        # Match by base name if needed
        obj_config = get_object_config(obj_name, objects_config)
        if obj_config is None:
            return False
        
        if self.has_contents(obj_name):
            chaining = obj_config.get('chaining', {})
            can_grasp_with_contents = chaining.get('can_grasp_with_contents', False)
            if not can_grasp_with_contents:
                return False
        
        return True
    
    def can_place_on_object(self, target_obj: str, objects_config: Dict) -> bool:
        """Check if something can be placed on/in the target object."""
        # Match by base name if needed
        obj_config = get_object_config(target_obj, objects_config)
        if obj_config is None:
            return False
        chaining = obj_config.get('chaining', {})
        max_items = chaining.get('max_items_on_top', 1)
        current_count = self.count_items_on_object(target_obj)
        return current_count < max_items
    
    def is_object_contained_in(self, inner_obj: str, container_obj: str) -> bool:
        """
        Check if inner_obj is contained within container_obj (recursively).
        
        This checks if inner_obj is directly on/in container_obj, or if it's
        contained within any object that's inside container_obj.
        
        Args:
            inner_obj: The object to check if it's contained
            container_obj: The container object to check within
        
        Returns:
            True if inner_obj is contained within container_obj, False otherwise
        """
        # Direct containment check
        items_in_container = self.get_items_on_object(container_obj)
        if inner_obj in items_in_container:
            return True
        
        # Recursive check: if any item in the container contains inner_obj
        for item in items_in_container:
            if self.is_object_contained_in(inner_obj, item):
                return True
        
        return False


def is_action_valid(
    task_spec: Dict,
    world_state: WorldState,
    objects_config: Dict,
    operators_config: Dict,
    is_first_action: bool = False
) -> bool:
    """
    Check if an action is valid given the current world state.
    
    Args:
        task_spec: The task specification to check
        world_state: Current state of the world
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
        is_first_action: Whether this is the first action in the chain
    
    Returns:
        True if the action is valid, False otherwise
    """
    task_type = task_spec['type']

    # PlaceIn: place into an already-open container; does not use action replay
    if task_type == 'PlaceIn':
        container = task_spec['obj1']
        obj_to_place = task_spec['obj2']
        # Container must already be open
        if container not in world_state.opened_objects:
            return False
        # Object must be graspable
        if not world_state.is_object_available_for_grasp(obj_to_place, objects_config):
            return False
        # Container must have space
        if not world_state.can_place_on_object(container, objects_config):
            return False
        return True

    # Check if this action requires action replay
    if task_type == 'On':
        variant = task_spec.get('variant', 'grasp')
        if variant == 'grasp':
            operator_name = 'can_grasp'
        else:
            operator_name = 'can_push'
    elif task_type == 'In':
        # In tasks have open as first step
        operator_name = 'can_open'
    elif task_type == 'Open':
        operator_name = 'can_open'
    elif task_type == 'Turnon':
        operator_name = 'can_turn_on'
    else:
        return False

    op_config = operators_config.get(operator_name, {})
    requires_replay = op_config.get('requires_action_replay', False)

    # Actions requiring replay must be first (action replay can only be used once)
    # Note: independent actions that require replay still must be first
    if requires_replay and not is_first_action:
        return False

    # If action replay has been used and this action requires replay, invalid
    if requires_replay and world_state.action_replay_used:
        return False

    # Type-specific checks
    if task_type == 'On':
        variant = task_spec.get('variant', 'grasp')
        obj1 = task_spec['obj1']  # object being moved
        obj2 = task_spec['obj2']  # target location
        
        if variant == 'grasp':
            # Check if object can be grasped
            if not world_state.is_object_available_for_grasp(obj1, objects_config):
                return False
        else:
            # Push - object must not have been placed
            if obj1 in world_state.placed_objects:
                return False
        
        # Check if target can receive items
        if not world_state.can_place_on_object(obj2, objects_config):
            return False
        
        # Prevent logical contradiction: can't place obj1 on obj2 if obj2 is already inside obj1
        # Example: can't put bowl on cream_cheese if cream_cheese is already in the bowl
        if world_state.is_object_contained_in(obj2, obj1):
            return False
        
        # Prevent logical contradiction: can't place obj1 on obj2 if obj1 is already inside obj2
        # Example: can't put cream_cheese on bowl if cream_cheese is already in the bowl
        if world_state.is_object_contained_in(obj1, obj2):
            return False
            
    elif task_type == 'In':
        container = task_spec['obj1']
        obj_to_place = task_spec['obj2']
        
        # Container must not already be opened (can't open twice)
        if container in world_state.opened_objects:
            return False
        
        # Object to place must be graspable
        if not world_state.is_object_available_for_grasp(obj_to_place, objects_config):
            return False
        
        # Container must have space
        if not world_state.can_place_on_object(container, objects_config):
            return False
            
    elif task_type == 'Open':
        obj = task_spec['obj1']
        # Can't open if already opened
        if obj in world_state.opened_objects:
            return False
            
    elif task_type == 'Turnon':
        obj = task_spec['obj1']
        # Can't turn on if already on
        if obj in world_state.turned_on_objects:
            return False
    
    return True


def apply_action(
    task_spec: Dict,
    world_state: WorldState,
    objects_config: Dict,
    operators_config: Dict
) -> WorldState:
    """
    Apply an action and return the new world state.
    
    Args:
        task_spec: The task specification to apply
        world_state: Current state of the world
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
    
    Returns:
        New world state after applying the action
    """
    new_state = world_state.copy()
    task_type = task_spec['type']
    
    # Track first action type
    if new_state.first_action_type is None:
        new_state.first_action_type = task_type
    
    # Check if this action requires replay
    if task_type == 'On':
        variant = task_spec.get('variant', 'grasp')
        operator_name = 'can_grasp' if variant == 'grasp' else 'can_push'
    elif task_type in ['In', 'Open']:
        operator_name = 'can_open'
    elif task_type == 'Turnon':
        operator_name = 'can_turn_on'
    elif task_type == 'PlaceIn':
        operator_name = None  # PlaceIn doesn't use action replay (container already open)
    else:
        operator_name = None
    
    if operator_name:
        op_config = operators_config.get(operator_name, {})
        requires_replay = op_config.get('requires_action_replay', False)
        # Mark action replay as used
        if requires_replay:
            new_state.action_replay_used = True
    
    # Apply type-specific state changes
    if task_type == 'On':
        obj1 = task_spec['obj1']  # object being moved
        obj2 = task_spec['obj2']  # target location
        
        # Mark object as placed
        new_state.placed_objects.add(obj1)
        
        # Add object to target's contents
        if obj2 not in new_state.objects_on_top:
            new_state.objects_on_top[obj2] = []
        new_state.objects_on_top[obj2].append(obj1)
        
    elif task_type == 'In':
        container = task_spec['obj1']
        obj_to_place = task_spec['obj2']
        
        # Mark container as opened
        new_state.opened_objects.add(container)
        
        # Mark object as placed
        new_state.placed_objects.add(obj_to_place)
        
        # Add object to container's contents
        if container not in new_state.objects_on_top:
            new_state.objects_on_top[container] = []
        new_state.objects_on_top[container].append(obj_to_place)
        
    elif task_type == 'PlaceIn':
        container = task_spec['obj1']
        obj_to_place = task_spec['obj2']

        # Mark object as placed (container is already open from a prior Open action)
        new_state.placed_objects.add(obj_to_place)

        # Add object to container's contents
        if container not in new_state.objects_on_top:
            new_state.objects_on_top[container] = []
        new_state.objects_on_top[container].append(obj_to_place)

    elif task_type == 'Open':
        obj = task_spec['obj1']
        new_state.opened_objects.add(obj)

    elif task_type == 'Turnon':
        obj = task_spec['obj1']
        new_state.turned_on_objects.add(obj)

    return new_state


class BDDLParser:
    def __init__(self, file_content: str):
        self.file_content = file_content
        self.objects_of_interest = self._parse_obj_of_interest()
        self.initial_states = self._parse_initial_states()
        self.all_objects = self._parse_all_objects()
        self.goal_conditions = self._parse_goal_conditions()

    def _find_outer_block_span(self, text: str, head: str):
        """Find the span of an outer block (handles nested parentheses)."""
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _parse_goal_conditions(self) -> List[Dict[str, str]]:
        """Parse goal conditions from the (:goal) section.
        
        Returns a list of goal predicates, each as a dict with:
        - 'type': 'On', 'In', 'Open', or 'Turnon'
        - 'obj1': first object (or container for In, target for On)
        - 'obj2': second object (object to place for In, source for On) - optional
        """
        goal_conditions = []
        
        # Find the goal block using the same approach as _find_outer_block_span
        goal_span = self._find_outer_block_span(self.file_content, "(:goal")
        if not goal_span:
            return goal_conditions
        
        start, end = goal_span
        # Extract content inside (:goal ... )
        goal_text = self.file_content[start:end]
        # Remove the (:goal and closing ) wrapper
        goal_text = goal_text[6:].strip()  # Remove "(:goal"
        if goal_text.endswith(')'):
            goal_text = goal_text[:-1].strip()
        
        # Parse (And ...) expressions - can contain single or multiple predicates
        # Handle both (And (On obj1 obj2)) and (And (On obj1 obj2) (In obj3 obj4))
        and_match = re.search(r"\(And\s+(.*)\)", goal_text, re.DOTALL)
        if and_match:
            # Extract all predicates inside And
            predicates_text = and_match.group(1)
            # Find all top-level predicates (not nested) in order
            # Match patterns like (On obj1 obj2), (In obj1 obj2), (Open obj), (Turnon obj)
            # Collect all matches with their positions to preserve order
            all_matches = []
            predicate_patterns = [
                (r"\(On\s+(\w+)\s+(\w+)\)", 'On'),
                (r"\(In\s+(\w+)\s+(\w+)\)", 'In'),
                (r"\(Open\s+(\w+)\)", 'Open'),
                (r"\(Turnon\s+(\w+)\)", 'Turnon'),
            ]
            
            for pattern, pred_type in predicate_patterns:
                for match in re.finditer(pattern, predicates_text):
                    all_matches.append((match.start(), pred_type, match))
            
            # Sort by position to preserve order
            all_matches.sort(key=lambda x: x[0])
            
            # Process matches in order
            for _, pred_type, match in all_matches:
                if pred_type == 'On':
                    goal_conditions.append({
                        'type': 'On',
                        'obj1': match.group(1),  # object being moved
                        'obj2': match.group(2)   # target location
                    })
                elif pred_type == 'In':
                    goal_conditions.append({
                        'type': 'In',
                        'obj1': match.group(2),  # container
                        'obj2': match.group(1)    # object to place
                    })
                elif pred_type == 'Open':
                    goal_conditions.append({
                        'type': 'Open',
                        'obj1': match.group(1)
                    })
                elif pred_type == 'Turnon':
                    goal_conditions.append({
                        'type': 'Turnon',
                        'obj1': match.group(1)
                    })
        else:
            # No And wrapper, try to match single predicate directly
            predicate_patterns = [
                (r"\(On\s+(\w+)\s+(\w+)\)", 'On'),
                (r"\(In\s+(\w+)\s+(\w+)\)", 'In'),
                (r"\(Open\s+(\w+)\)", 'Open'),
                (r"\(Turnon\s+(\w+)\)", 'Turnon'),
            ]
            
            for pattern, pred_type in predicate_patterns:
                match = re.search(pattern, goal_text)
                if match:
                    if pred_type == 'On':
                        goal_conditions.append({
                            'type': 'On',
                            'obj1': match.group(1),
                            'obj2': match.group(2)
                        })
                    elif pred_type == 'In':
                        goal_conditions.append({
                            'type': 'In',
                            'obj1': match.group(2),
                            'obj2': match.group(1)
                        })
                    elif pred_type == 'Open':
                        goal_conditions.append({
                            'type': 'Open',
                            'obj1': match.group(1)
                        })
                    elif pred_type == 'Turnon':
                        goal_conditions.append({
                            'type': 'Turnon',
                            'obj1': match.group(1)
                        })
                    break
        
        return goal_conditions

    def _parse_obj_of_interest(self) -> List[str]:
        obj_pattern = r'\(:obj_of_interest(.*?)\)'
        obj_match = re.search(obj_pattern, self.file_content, re.DOTALL)
        if not obj_match:
            return []
        obj_content = obj_match.group(1)
        # Match identifiers: sequences of word characters and underscores
        # This matches both object names (plate_1) and region names (main_table_stove_front_region)
        objects = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', obj_content)
        return objects

    def _parse_initial_states(self) -> Dict[str, str]:
        initial_states = {}
        init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", self.file_content, re.S)
        if not init_block_match:
            return initial_states
        init_text = init_block_match.group(1)
        for match in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_text):
            obj, region = match.groups()
            initial_states[obj] = region
        # Also parse In predicates
        for match in re.finditer(r"\(In\s+(\w+)\s+(\w+)\)", init_text):
            obj, container = match.groups()
            initial_states[obj] = container
        return initial_states
    
    def get_initial_location(self, obj_name: str) -> Optional[str]:
        """Get the initial location of an object from the init section.
        
        Returns the object/region that obj_name is On or In, or None if not found.
        """
        return self.initial_states.get(obj_name)
    
    def get_initial_location_relation(self, obj_name: str) -> Optional[Tuple[str, str]]:
        """Get the initial location and relation type of an object from the init section.
        
        Returns a tuple of (location, relation_type) where relation_type is 'On' or 'In',
        or None if not found.
        """
        init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", self.file_content, re.S)
        if not init_block_match:
            return None
        init_text = init_block_match.group(1)
        # Check for On first
        on_match = re.search(rf"\(On\s+{obj_name}\s+(\w+)\)", init_text)
        if on_match:
            return (on_match.group(1), 'On')
        # Check for In
        in_match = re.search(rf"\(In\s+{obj_name}\s+(\w+)\)", init_text)
        if in_match:
            return (in_match.group(1), 'In')
        return None

    def _parse_all_objects(self) -> List[str]:
        """Parse all objects from the (:objects section and (:obj_of_interest section"""
        objects = []
        
        # Parse from (:objects section
        obj_pattern = r'\(:objects(.*?)\)'
        obj_match = re.search(obj_pattern, self.file_content, re.DOTALL)
        if obj_match:
            obj_content = obj_match.group(1)
            # Extract object names (format: object_name - object_type)
            # Match word_number pattern
            objects.extend(re.findall(r'(\w+_\d+)', obj_content))
        
        # Also include objects from obj_of_interest (includes regions)
        objects.extend(self.objects_of_interest)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_objects = []
        for obj in objects:
            if obj not in seen:
                seen.add(obj)
                unique_objects.append(obj)
        
        return unique_objects


def _parse_open_turnon_from_bddl(bddl_content: str) -> frozenset:
    """Parse Open and Turnon predicates from BDDL :init section.
    Returns frozenset of (predicate_type, obj_name), e.g. frozenset({("Open", "wooden_cabinet_1_middle_region")}).
    Used for exact init matching in match_init_states_from (does not modify BDDLParser.initial_states).
    """
    result = set()
    init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", bddl_content, re.S)
    if not init_block_match:
        return frozenset()
    init_text = init_block_match.group(1)
    for match in re.finditer(r"\(Open\s+(\w+)\)", init_text):
        result.add(("Open", match.group(1)))
    for match in re.finditer(r"\(Turnon\s+(\w+)\)", init_text):
        result.add(("Turnon", match.group(1)))
    return frozenset(result)


def parse_regions_from_bddl(bddl_content: str) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
    """Parse regions and their ranges from BDDL content.
    
    Args:
        bddl_content: The BDDL file content as a string
        
    Returns:
        Dictionary mapping region_name -> (x1, y1, x2, y2) or None if no ranges defined
    """
    regions = {}
    
    # Find the :regions section by finding balanced parentheses
    regions_start = bddl_content.find("(:regions")
    if regions_start == -1:
        return regions
    
    # Find the matching closing parenthesis for the entire :regions block
    i = regions_start + len("(:regions")
    depth = 1
    regions_end = i
    while regions_end < len(bddl_content) and depth > 0:
        if bddl_content[regions_end] == '(':
            depth += 1
        elif bddl_content[regions_end] == ')':
            depth -= 1
        regions_end += 1
    
    if depth != 0:
        # Unbalanced parentheses, return empty
        return regions
    
    # Extract the content inside the :regions block (excluding the outer parentheses)
    regions_content = bddl_content[regions_start + len("(:regions"):regions_end - 1]
    
    # Parse each region definition by finding balanced parentheses
    # Pattern: (region_name ... )
    i = 0
    while i < len(regions_content):
        # Skip whitespace
        while i < len(regions_content) and regions_content[i].isspace():
            i += 1
        if i >= len(regions_content):
            break
        
        # Look for opening parenthesis
        if regions_content[i] != '(':
            i += 1
            continue
        
        # Find the region name (first word after opening paren)
        region_start = i
        i += 1
        region_name_match = re.match(r"(\w+)", regions_content[i:])
        if not region_name_match:
            i += 1
            continue
        
        region_name = region_name_match.group(1)
        i += len(region_name)
        
        # Find the matching closing parenthesis for this region block
        depth = 1
        region_end = i
        while region_end < len(regions_content) and depth > 0:
            if regions_content[region_end] == '(':
                depth += 1
            elif regions_content[region_end] == ')':
                depth -= 1
            region_end += 1
        
        if depth == 0:
            # Extract the region block
            region_block = regions_content[region_start:region_end]
            
            # Look for :ranges within this region block
            # Pattern: (:ranges ((x1 y1 x2 y2)))
            ranges_match = re.search(r"\(:ranges\s*\(\s*\(([^)]+)\)\s*\)", region_block, re.DOTALL)
            if ranges_match:
                ranges_str = ranges_match.group(1)
                # Extract the four numbers: x1 y1 x2 y2
                numbers = re.findall(r"(-?\d+\.?\d*)", ranges_str)
                if len(numbers) >= 4:
                    try:
                        x1, y1, x2, y2 = float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3])
                        regions[region_name] = (x1, y1, x2, y2)
                    except ValueError:
                        regions[region_name] = None
                else:
                    regions[region_name] = None
            else:
                # Region exists but has no ranges defined
                regions[region_name] = None
            
            i = region_end
        else:
            # Unbalanced parentheses, skip
            i += 1
    
    return regions


def calculate_region_area(ranges: Optional[Tuple[float, float, float, float]]) -> Optional[float]:
    """Calculate the area of a region from its ranges.
    
    Args:
        ranges: Tuple of (x1, y1, x2, y2) defining the rectangular region bounds, or None
        
    Returns:
        Area of the region (|x2 - x1| * |y2 - y1|), or None if ranges is None
    """
    if ranges is None:
        return None
    x1, y1, x2, y2 = ranges
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width * height


def calculate_scaled_region_bounds(original_ranges: Tuple[float, float, float, float], min_area: float) -> Tuple[float, float, float, float]:
    """Calculate new region bounds that are centered at the same location, have the same aspect ratio,
    but are scaled up to meet the minimum area requirement.
    
    Args:
        original_ranges: Tuple of (x1, y1, x2, y2) defining the original rectangular region bounds
        min_area: Minimum area required for the new region
        
    Returns:
        Tuple of (new_x1, new_y1, new_x2, new_y2) for the scaled region
    """
    x1, y1, x2, y2 = original_ranges
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    original_area = width * height
    
    if original_area >= min_area:
        return original_ranges
    
    # Calculate aspect ratio
    if height == 0:
        aspect_ratio = 1.0
    else:
        aspect_ratio = width / height
    
    # Calculate new dimensions maintaining aspect ratio
    # new_area = new_width * new_height = min_area
    # new_width / new_height = aspect_ratio
    # Solving: new_height = sqrt(min_area / aspect_ratio), new_width = aspect_ratio * new_height
    if aspect_ratio > 0:
        new_height = (min_area / aspect_ratio) ** 0.5
        new_width = aspect_ratio * new_height
    else:
        new_width = min_area ** 0.5
        new_height = new_width
    
    # Calculate center point
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    
    # Calculate new bounds centered at the same location
    new_x1 = center_x - new_width / 2.0
    new_x2 = center_x + new_width / 2.0
    new_y1 = center_y - new_height / 2.0
    new_y2 = center_y + new_height / 2.0
    
    return (new_x1, new_y1, new_x2, new_y2)


def parse_all_regions_from_bddl(bddl_content: str) -> List[str]:
    """Parse all region names from the :regions section.
    
    Regions are named as {target}_{region_name} (e.g., main_table_stove_front_region).
    
    Args:
        bddl_content: The BDDL file content as a string
        
    Returns:
        List of all region names found in the :regions section
    """
    regions = []
    
    # Find the :regions section
    regions_start = bddl_content.find("(:regions")
    if regions_start == -1:
        return regions
    
    # Find the matching closing parenthesis for the entire :regions block
    i = regions_start + len("(:regions")
    depth = 1
    regions_end = i
    while regions_end < len(bddl_content) and depth > 0:
        if bddl_content[regions_end] == '(':
            depth += 1
        elif bddl_content[regions_end] == ')':
            depth -= 1
        regions_end += 1
    
    if depth != 0:
        return regions
    
    # Extract the content inside the :regions block
    regions_content = bddl_content[regions_start + len("(:regions"):regions_end - 1]
    
    # Parse each region definition
    i = 0
    while i < len(regions_content):
        # Skip whitespace
        while i < len(regions_content) and regions_content[i].isspace():
            i += 1
        if i >= len(regions_content):
            break
        
        # Look for opening parenthesis
        if regions_content[i] != '(':
            i += 1
            continue
        
        # Find the region name (first word after opening paren)
        region_start = i
        i += 1
        region_name_match = re.match(r"(\w+)", regions_content[i:])
        if not region_name_match:
            i += 1
            continue
        
        region_name = region_name_match.group(1)
        i += len(region_name)
        
        # Find the matching closing parenthesis for this region block
        depth = 1
        region_end = i
        while region_end < len(regions_content) and depth > 0:
            if regions_content[region_end] == '(':
                depth += 1
            elif regions_content[region_end] == ')':
                depth -= 1
            region_end += 1
        
        if depth == 0:
            # Extract the region block
            region_block = regions_content[region_start:region_end]
            
            # Extract target from (:target ...)
            target_match = re.search(r"\(:target\s+(\w+)", region_block)
            if target_match:
                target = target_match.group(1)
                # Region name is {target}_{region_name}
                full_region_name = f"{target}_{region_name}"
                regions.append(full_region_name)
            
            i = region_end
        else:
            # Unbalanced parentheses, skip
            i += 1
    
    return regions


def get_region_target(bddl_content: str, base_region_name: str) -> str:
    """Extract the target object for a region from the BDDL content.
    
    Args:
        bddl_content: The BDDL file content as a string
        base_region_name: The base name of the region (without prefix)
        
    Returns:
        Target object name (default: "main_table" if not found)
    """
    # Find the :regions section
    regions_start = bddl_content.find("(:regions")
    if regions_start == -1:
        return "main_table"
    
    # Find the matching closing parenthesis for the entire :regions block
    i = regions_start + len("(:regions")
    depth = 1
    regions_end = i
    while regions_end < len(bddl_content) and depth > 0:
        if bddl_content[regions_end] == '(':
            depth += 1
        elif bddl_content[regions_end] == ')':
            depth -= 1
        regions_end += 1
    
    if depth != 0:
        return "main_table"
    
    regions_content = bddl_content[regions_start + len("(:regions"):regions_end - 1]
    
    # Find the region definition
    region_pattern = rf"\({re.escape(base_region_name)}.*?\)"
    region_match = re.search(region_pattern, regions_content, re.DOTALL)
    if not region_match:
        return "main_table"
    
    region_block = region_match.group(0)
    
    # Extract target from (:target ...)
    target_match = re.search(r"\(:target\s+(\w+)", region_block)
    if target_match:
        return target_match.group(1)
    
    return "main_table"


def insert_region_into_bddl(bddl_content: str, region_name: str, ranges: Tuple[float, float, float, float], target: str = "main_table") -> str:
    """Insert a new region definition into the BDDL file's :regions section.
    
    Args:
        bddl_content: The BDDL file content as a string
        region_name: Name of the new region to insert
        ranges: Tuple of (x1, y1, x2, y2) defining the region bounds
        target: Target object for the region (default: "main_table")
        
    Returns:
        Modified BDDL content with the new region inserted
    """
    x1, y1, x2, y2 = ranges
    
    # Find the :regions section
    regions_start = bddl_content.find("(:regions")
    if regions_start == -1:
        raise ValueError("Could not find :regions section in BDDL content")
    
    # Find the matching closing parenthesis for the entire :regions block
    i = regions_start + len("(:regions")
    depth = 1
    regions_end = i
    while regions_end < len(bddl_content) and depth > 0:
        if bddl_content[regions_end] == '(':
            depth += 1
        elif bddl_content[regions_end] == ')':
            depth -= 1
        regions_end += 1
    
    if depth != 0:
        raise ValueError("Could not find balanced parentheses for :regions section")
    
    # Find the insertion point (before the closing parenthesis of :regions)
    # We want to insert right before the closing ')' of the :regions block
    insertion_point = regions_end - 1
    
    # Strip any trailing whitespace before the closing parenthesis to ensure clean insertion
    # Look backwards to find where the actual content ends
    content_before = bddl_content[:insertion_point].rstrip()
    
    # Create the new region definition with proper indentation (6 spaces to match other regions)
    new_region = f"""      ({region_name}
          (:target {target})
          (:ranges (
              ({x1} {y1} {x2} {y2})
            )
          )
      )
"""
    
    # Insert the new region with a newline before it, then restore the closing parenthesis and newline
    # This ensures proper formatting
    modified_content = content_before + "\n" + new_region + bddl_content[insertion_point:]
    
    return modified_content


def update_goal_condition_region(bddl_content: str, old_region_name: str, new_region_name: str) -> str:
    """Update a region name in the goal condition.
    
    Args:
        bddl_content: The BDDL file content as a string
        old_region_name: The old region name to replace (may have prefixes)
        new_region_name: The new region name to use (may have prefixes)
        
    Returns:
        Modified BDDL content with the region name updated in goal conditions
    """
    # Find the :goal section by finding balanced parentheses
    goal_start = bddl_content.find("(:goal")
    if goal_start == -1:
        return bddl_content
    
    # Find the matching closing parenthesis for the :goal block
    i = goal_start + len("(:goal")
    depth = 1
    goal_end = i
    while goal_end < len(bddl_content) and depth > 0:
        if bddl_content[goal_end] == '(':
            depth += 1
        elif bddl_content[goal_end] == ')':
            depth -= 1
        goal_end += 1
    
    if depth != 0:
        return bddl_content
    
    # Extract goal content (inside the parentheses)
    goal_content_start = goal_start + len("(:goal")
    goal_content_end = goal_end - 1
    goal_content = bddl_content[goal_content_start:goal_content_end]
    
    # Replace the old region name with the new one in On predicates
    # Pattern: (On obj old_region_name) -> (On obj new_region_name)
    updated_goal = re.sub(
        rf"\(On\s+(\w+)\s+{re.escape(old_region_name)}\)",
        rf"(On \1 {new_region_name})",
        goal_content
    )
    
    # Reconstruct the BDDL content
    modified_content = bddl_content[:goal_content_start] + updated_goal + bddl_content[goal_content_end:]
    
    return modified_content


def update_execution_steps_region(execution_steps: List[str], old_region_name: str, new_region_name: str) -> List[str]:
    """Update a region name in execution steps.
    
    Args:
        execution_steps: List of execution step strings (e.g., ["Place obj1 region_name"])
        old_region_name: The old region name to replace
        new_region_name: The new region name to use
        
    Returns:
        Updated list of execution steps
    """
    updated_steps = []
    for step in execution_steps:
        # Replace region name in Place steps: "Place obj old_region" -> "Place obj new_region"
        if step.startswith("Place "):
            parts = step.split()
            if len(parts) >= 3 and parts[2] == old_region_name:
                updated_steps.append(f"{parts[0]} {parts[1]} {new_region_name}")
            else:
                updated_steps.append(step)
        else:
            updated_steps.append(step)
    return updated_steps


def resize_regions_for_place_operations(
    bddl_content: str,
    execution_steps: List[str],
    num_replayed_steps: int,
    min_region_size: float,
    regions: Dict[str, Optional[Tuple[float, float, float, float]]]
) -> Tuple[str, List[str], Dict[str, str]]:
    """Resize regions that are too small for place operations occurring after actions_from steps.
    
    This function:
    1. Identifies regions used in place operations after actions_from steps
    2. Creates new scaled regions for regions that are too small
    3. Updates the BDDL content with new regions
    4. Updates goal conditions to use new region names
    5. Updates execution steps to use new region names
    
    Args:
        bddl_content: The BDDL file content as a string
        execution_steps: List of execution steps for this task
        num_replayed_steps: Number of steps that will be replayed from actions_from
        min_region_size: Minimum area required for regions used in place operations
        regions: Dictionary of parsed regions from the BDDL
        
    Returns:
        Tuple of (modified_bddl_content, updated_execution_steps, region_name_mapping)
        where region_name_mapping maps old_region_name -> new_region_name
    """
    modified_bddl = bddl_content
    updated_steps = execution_steps.copy()
    region_name_mapping = {}  # Maps old region name (with prefix) -> new region name (with prefix)
    
    # Parse goal conditions to find "On" predicates
    parser = BDDLParser(modified_bddl)
    goal_conditions = parser.goal_conditions
    
    # Find which execution steps involve placing on regions (after actions_from)
    steps_after_replay = execution_steps[num_replayed_steps:]
    
    # Track which regions are used in steps after replay
    regions_to_resize = {}  # Maps old_region_name (with prefix) -> (base_region_name, ranges, step_index)
    
    for step_idx, step in enumerate(steps_after_replay):
        if step.startswith("Place "):
            parts = step.split()
            if len(parts) >= 3:
                obj1 = parts[1]  # object being placed
                obj2 = parts[2]  # target location (could be region or object)
                
                # Check if obj2 is a region
                region_name = None
                region_ranges = None
                
                if obj2 in regions:
                    region_name = obj2
                    region_ranges = regions[obj2]
                else:
                    # Try to match by checking if obj2 ends with a region name
                    # Sort by length (longest first) to match the most specific region name
                    sorted_regions = sorted(regions.items(), key=lambda x: len(x[0]), reverse=True)
                    for base_region_name, ranges in sorted_regions:
                        if obj2 == base_region_name or obj2.endswith('_' + base_region_name):
                            region_name = base_region_name
                            region_ranges = ranges
                            break
                
                if region_name is not None and region_ranges is not None:
                    # This is a place on region operation
                    area = calculate_region_area(region_ranges)
                    if area is not None and area < min_region_size:
                        # Store this region for resizing (use obj2 as the key to preserve prefix)
                        if obj2 not in regions_to_resize:
                            regions_to_resize[obj2] = (region_name, region_ranges, num_replayed_steps + step_idx)
    
    # Resize each region that needs it
    for old_region_name_with_prefix, (base_region_name, ranges, step_idx) in regions_to_resize.items():
        # Calculate new bounds
        new_ranges = calculate_scaled_region_bounds(ranges, min_region_size)
        
        # Generate new region name (add _resized suffix to base name)
        new_base_region_name = f"{base_region_name}_resized"
        
        # Determine the prefix for the new region name
        if old_region_name_with_prefix == base_region_name:
            new_region_name_with_prefix = new_base_region_name
        else:
            # Extract prefix (everything before the base region name)
            prefix = old_region_name_with_prefix[:-len(base_region_name)]
            if prefix.endswith('_'):
                new_region_name_with_prefix = prefix + new_base_region_name
            else:
                new_region_name_with_prefix = prefix + '_' + new_base_region_name
        
        # Determine target from original region
        target = get_region_target(modified_bddl, base_region_name)
        
        # Insert new region into BDDL
        modified_bddl = insert_region_into_bddl(modified_bddl, new_base_region_name, new_ranges, target)
        
        # Update goal condition
        modified_bddl = update_goal_condition_region(modified_bddl, old_region_name_with_prefix, new_region_name_with_prefix)
        
        # Update execution steps
        updated_steps = update_execution_steps_region(updated_steps, old_region_name_with_prefix, new_region_name_with_prefix)
        
        # Store mapping
        region_name_mapping[old_region_name_with_prefix] = new_region_name_with_prefix
    
    return modified_bddl, updated_steps, region_name_mapping


def extract_init_section(bddl_content: str) -> str:
    """Extract the (:init ...) section from BDDL content.
    
    Returns the init section content as a normalized string for comparison.
    """
    init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", bddl_content, re.S)
    if not init_block_match:
        return ""
    init_text = init_block_match.group(1)
    # Normalize whitespace and extract all On/In predicates
    init_predicates = []
    for match in re.finditer(r"\((On|In)\s+(\w+)\s+(\w+)\)", init_text):
        relation, obj, location = match.groups()
        init_predicates.append(f"({relation} {obj} {location})")
    # Sort to ensure consistent comparison regardless of order
    return " ".join(sorted(init_predicates))


def are_bddl_functionally_equivalent(bddl_content1: str, bddl_content2: str) -> Tuple[bool, bool]:
    """Compare the init and goal sections of two BDDL files for functional equivalence.
    
    Uses generic object names (base names without instance numbers) to check equivalence.
    Compares:
        - Init: Dictionary mapping generic location -> generic object name
        - Goal: Set of (generic_object, generic_location) tuples
    
    Returns:
        Tuple of (functional_equivalent, instance_level_match):
        - functional_equivalent: True if functionally equivalent (generic match), False otherwise
        - instance_level_match: True if exact match on instance level, False otherwise
    """
    parser1 = BDDLParser(bddl_content1)
    parser2 = BDDLParser(bddl_content2)
    
    # Check if textually identical (instance-level match)
    init1 = extract_init_section(bddl_content1)
    init2 = extract_init_section(bddl_content2)
    goal1 = parser1.goal_conditions
    goal2 = parser2.goal_conditions
    
    instance_level_match = (init1 == init2 and goal1 == goal2)
    
    # Check functional equivalence using generic object names
    functional_equivalent = _are_functionally_equivalent(parser1, parser2)
    
    return (functional_equivalent, instance_level_match)


def _strip_resized_suffix(name: str) -> str:
    """Strip the _resized suffix from a name if present.
    
    This is used to normalize region names for functional equivalence checking,
    since resized regions are functionally equivalent to their original regions.
    
    Args:
        name: The name to strip (e.g., "between_plate_ramekin_region_resized")
        
    Returns:
        The name with _resized suffix removed (e.g., "between_plate_ramekin_region")
    """
    if name.endswith('_resized'):
        return name[:-8]  # Remove '_resized' (8 characters)
    return name


def _are_functionally_equivalent(parser1: BDDLParser, parser2: BDDLParser) -> bool:
    """Check if two BDDL parsers represent functionally equivalent tasks.
    
    Compares using generic object names (base names without instance numbers).
    Also strips _resized suffix from region names to treat resized regions as equivalent.
    """
    # Build init state set: set of (generic_location, generic_object) tuples
    def build_init_set(parser: BDDLParser) -> Set[Tuple[str, str]]:
        """Build set of (generic_location, generic_object) tuples."""
        init_set = set()
        for obj, location in parser.initial_states.items():
            generic_obj = get_base_object_name(obj)
            generic_location = get_base_object_name(location)
            # Strip _resized suffix from location names
            generic_location = _strip_resized_suffix(generic_location)
            init_set.add((generic_location, generic_obj))
        return init_set
    
    init1_set = build_init_set(parser1)
    init2_set = build_init_set(parser2)
    
    # Check if init sets match
    if init1_set != init2_set:
        return False
    
    # Build goal conditions: set of (generic_object, generic_initial_location, generic_target_location, type) tuples
    def build_goal_set(parser: BDDLParser) -> Set[Tuple[str, str, str, str]]:
        """Build set of goal conditions using generic object names.
        
        Includes the initial location of the object being moved to distinguish tasks that move
        different instances of the same object type from different starting locations.
        
        For On: (generic_object, generic_initial_location, generic_target_location, goal_type)
               where obj1 is object being moved, obj2 is target location
        For In: (generic_object, generic_initial_location, generic_target_location, goal_type)
               where obj2 is object being placed, obj1 is container
        For Open/Turnon: (generic_object, generic_initial_location, '', goal_type)
                       where obj1 is the object
        """
        goal_set = set()
        for goal in parser.goal_conditions:
            goal_type = goal['type']
            
            if goal_type == 'On':
                # obj1 is object being moved, obj2 is target location
                obj_being_moved = goal['obj1']
                generic_obj = get_base_object_name(obj_being_moved)
                generic_target_location = get_base_object_name(goal['obj2'])
                # Strip _resized suffix from target location
                generic_target_location = _strip_resized_suffix(generic_target_location)
                # Get initial location of the object being moved
                initial_location = parser.initial_states.get(obj_being_moved)
                if initial_location is None:
                    # Object doesn't have an initial location (shouldn't happen, but handle gracefully)
                    generic_initial_location = ''
                else:
                    generic_initial_location = get_base_object_name(initial_location)
                    # Strip _resized suffix from initial location
                    generic_initial_location = _strip_resized_suffix(generic_initial_location)
                goal_set.add((generic_obj, generic_initial_location, generic_target_location, goal_type))
            elif goal_type == 'In':
                # obj1 is container, obj2 is object to place
                obj_being_moved = goal['obj2']
                generic_obj = get_base_object_name(obj_being_moved)
                generic_target_location = get_base_object_name(goal['obj1'])  # container
                # Strip _resized suffix from target location (container)
                generic_target_location = _strip_resized_suffix(generic_target_location)
                # Get initial location of the object being moved
                initial_location = parser.initial_states.get(obj_being_moved)
                if initial_location is None:
                    # Object doesn't have an initial location (shouldn't happen, but handle gracefully)
                    generic_initial_location = ''
                else:
                    generic_initial_location = get_base_object_name(initial_location)
                    # Strip _resized suffix from initial location
                    generic_initial_location = _strip_resized_suffix(generic_initial_location)
                goal_set.add((generic_obj, generic_initial_location, generic_target_location, goal_type))
            else:
                # Open or Turnon: obj1 is the object
                obj = goal['obj1']
                generic_obj = get_base_object_name(obj)
                # Get initial location of the object
                initial_location = parser.initial_states.get(obj)
                if initial_location is None:
                    # Object doesn't have an initial location (shouldn't happen, but handle gracefully)
                    generic_initial_location = ''
                else:
                    generic_initial_location = get_base_object_name(initial_location)
                    # Strip _resized suffix from initial location
                    generic_initial_location = _strip_resized_suffix(generic_initial_location)
                goal_set.add((generic_obj, generic_initial_location, '', goal_type))
        return goal_set
    
    goal1_set = build_goal_set(parser1)
    goal2_set = build_goal_set(parser2)
    
    # Check if goal sets match
    return goal1_set == goal2_set


class BDDLPerturbator:
    def __init__(self, split_name: str, base_task_filename: str, new_lang: Optional[str]=None, new_goal: Optional[str]=None, new_objs_of_interest: Optional[list]=None, bddl_files_base: Optional[str]=None, bddl_content: Optional[str]=None):
        self.split_name = split_name
        self.base_task_filename = base_task_filename
        self.new_lang = new_lang
        self.new_goal = new_goal
        self.new_objs_of_interest = new_objs_of_interest
        
        # load the bddl parser for the base task
        # If bddl_content is provided, use it directly (for swapped versions)
        # Otherwise, load from file
        if bddl_content is not None:
            content = bddl_content
        else:
            # Use provided bddl_files_base if given, otherwise fall back to default
            if bddl_files_base is None:
                bddl_files_base = get_libero_path('bddl_files')
            bddl_path = os.path.join(bddl_files_base, split_name, base_task_filename + '.bddl')

            with open(bddl_path, "r", encoding="utf-8") as f:
                content = f.read()

        self.parser = BDDLParser(content)

    def _find_language_inner_span(self, text: str):
        m = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)
        if not m:
            return None
        return (m.start(1), m.end(1), m.group(1))

    def _find_outer_block_span(self, text: str, head: str):
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _replace_language(self, text: str, new_lang: str) -> str:
        span = self._find_language_inner_span(text)
        if not span:
            return text
        s, e, old = span
        return text[:s] + new_lang + text[e:]

    def _replace_goal(self, text: str, new_goal_expr: str) -> str:
        span = self._find_outer_block_span(text, "(:goal")
        if not span:
            return text
        start, end = span
        replacement = "(:goal\n    " + new_goal_expr + "\n  )"
        return text[:start] + replacement + text[end:]

    def _replace_obj_of_interest(self, text: str, new_objs: list) -> str:
        span = self._find_outer_block_span(text, "(:obj_of_interest")
        if not span:
            return text
        start, end = span
        replacement = "(:obj_of_interest\n"
        for obj in new_objs:
            replacement += f"    {obj}\n"
        replacement += "  )"
        return text[:start] + replacement + text[end:]
    
    def _replace_init_move(self, text: str, from_object: str, to_location: str, object_instance: Optional[str] = None, objects_config: Optional[Dict] = None) -> str:
        """
        Move an object's initial location in the :init section.
        
        Args:
            text: BDDL file content
            from_object: Base name of object to move (e.g., 'akita_black_bowl')
            to_location: Base name of target location (e.g., 'plate')
            object_instance: Optional specific object instance to move (e.g., 'akita_black_bowl_1').
                           If None, finds the first matching instance by base name.
            objects_config: Optional objects configuration to determine correct relation type (On vs In)
        
        Returns:
            Modified BDDL content with object moved to new location
        """
        # Parse current initial states to find the object instance
        parser = BDDLParser(text)
        initial_states = parser.initial_states
        
        # If object_instance is provided, use it; otherwise find first matching instance
        if object_instance is None:
            # Find object instance(s) that match from_object (by base name)
            for obj_name, location in initial_states.items():
                obj_base = get_base_object_name(obj_name)
                if obj_base == from_object:
                    object_instance = obj_name
                    break
        
        if object_instance is None:
            # Object not found in initial states, return unchanged
            return text
        
        # Verify that the provided object_instance matches the base name
        if get_base_object_name(object_instance) != from_object:
            # Provided instance doesn't match base name, return unchanged
            return text
        
        # Find the target location instance (by base name)
        # Check in all_objects first, then check all regions from :regions section,
        # then check in initial_states (locations where objects are placed)
        target_location_instance = None
        
        # Check all objects from :objects and :obj_of_interest
        all_objects = parser.all_objects
        for obj_name in all_objects:
            obj_base = get_base_object_name(obj_name)
            if obj_base == to_location:
                target_location_instance = obj_name
                break
        
        # If not found in all_objects, check all regions from :regions section
        if target_location_instance is None:
            all_regions = parse_all_regions_from_bddl(text)
            for region_name in all_regions:
                region_base = get_base_object_name(region_name)
                if region_base == to_location:
                    target_location_instance = region_name
                    break
        
        # If still not found, check in initial_states (locations where objects are placed)
        # This catches any edge cases
        if target_location_instance is None:
            for location in initial_states.values():
                location_base = get_base_object_name(location)
                if location_base == to_location:
                    target_location_instance = location
                    break
        
        if target_location_instance is None:
            # Target location not found, return unchanged
            return text
        
        # Get current location
        current_location = initial_states.get(object_instance)
        if current_location is None:
            return text
        
        # Determine the correct relation type (On vs In) based on target location
        # If objects_config is provided, check what operators the target location supports
        use_in_relation = False
        if objects_config is not None:
            target_config_key = match_object_to_config(target_location_instance, objects_config)
            if target_config_key is not None:
                target_obj_config = objects_config[target_config_key]
                operators = target_obj_config.get('operators', {})
                if isinstance(operators, dict):
                    # If target has can_place_in, use In relation; otherwise use On
                    use_in_relation = 'can_place_in' in operators
                elif isinstance(operators, list):
                    # Old format: list of operator names
                    use_in_relation = 'can_place_in' in operators
        
        # Find the :init section
        init_span = self._find_outer_block_span(text, "(:init")
        if not init_span:
            return text
        
        start, end = init_span
        init_content = text[start:end]
        
        # Find and replace the predicate - use the determined relation type
        pattern_on = rf"\(On\s+{re.escape(object_instance)}\s+{re.escape(current_location)}\)"
        pattern_in = rf"\(In\s+{re.escape(object_instance)}\s+{re.escape(current_location)}\)"
        
        if use_in_relation:
            replacement = f"(In {object_instance} {target_location_instance})"
            # Try to find existing In predicate first, then On
            if re.search(pattern_in, init_content):
                init_content = re.sub(pattern_in, replacement, init_content)
            elif re.search(pattern_on, init_content):
                init_content = re.sub(pattern_on, replacement, init_content)
            else:
                # Predicate not found, return unchanged
                return text
        else:
            replacement = f"(On {object_instance} {target_location_instance})"
            # Try to find existing On predicate first, then In
            if re.search(pattern_on, init_content):
                init_content = re.sub(pattern_on, replacement, init_content)
            elif re.search(pattern_in, init_content):
                init_content = re.sub(pattern_in, replacement, init_content)
            else:
                # Predicate not found, return unchanged
                return text
        
        # Reconstruct the full text
        return text[:start] + init_content + text[end:]
    
    def _apply_additional_modifications(self, text: str, additional_modifications: Dict) -> str:
        """
        Apply additional modifications to BDDL content, such as adding init state predicates.
        
        Args:
            text: BDDL file content
            additional_modifications: Dict with modification specifications, e.g.:
                {
                    'init_states': [
                        ['Open', 'wooden_cabinet_1_top_region'],
                        ...
                    ]
                }
        
        Returns:
            Modified BDDL content with additional modifications applied
        """
        if not additional_modifications:
            return text
        
        # Find the :init section
        init_span = self._find_outer_block_span(text, "(:init")
        if not init_span:
            return text
        
        start, end = init_span
        init_content = text[start:end]
        
        # Apply init_states modifications
        init_states_mods = additional_modifications.get('init_states', [])
        for mod in init_states_mods:
            if not isinstance(mod, list) or len(mod) < 2:
                print(f"WARNING: Invalid init_states modification format: {mod}. Expected [predicate_type, object_name, ...]")
                continue
            
            predicate_type = mod[0]
            obj_name_spec = mod[1]  # May be base name or instance name
            
            # Try to find the actual object/region instance in the BDDL
            # First check if it exists as-is
            parser = BDDLParser(text)
            all_objects_and_regions = set(parser.all_objects)
            all_regions = parse_all_regions_from_bddl(text)
            all_objects_and_regions.update(all_regions)
            # Also check initial_states for locations
            for location in parser.initial_states.values():
                all_objects_and_regions.add(location)
            
            obj_name = obj_name_spec
            obj_name_base = get_base_object_name(obj_name_spec)
            
            # Try to find matching instance
            if obj_name_spec not in all_objects_and_regions:
                # Try to find by base name
                for obj in all_objects_and_regions:
                    if get_base_object_name(obj) == obj_name_base:
                        obj_name = obj
                        break
                else:
                    # Not found, use the specified name as-is (might be added later)
                    obj_name = obj_name_spec
            
            # Construct the predicate string
            if predicate_type == 'Open':
                # Format: (Open obj_name)
                predicate = f"(Open {obj_name})"
            elif predicate_type == 'On':
                # Format: (On obj1 obj2)
                if len(mod) < 3:
                    print(f"WARNING: On predicate requires 3 arguments, got {len(mod)}: {mod}")
                    continue
                obj2_spec = mod[2]
                # Try to find obj2 instance
                obj2 = obj2_spec
                obj2_base = get_base_object_name(obj2_spec)
                if obj2_spec not in all_objects_and_regions:
                    for obj in all_objects_and_regions:
                        if get_base_object_name(obj) == obj2_base:
                            obj2 = obj
                            break
                    else:
                        obj2 = obj2_spec
                predicate = f"(On {obj_name} {obj2})"
            elif predicate_type == 'In':
                # Format: (In obj1 obj2)
                if len(mod) < 3:
                    print(f"WARNING: In predicate requires 3 arguments, got {len(mod)}: {mod}")
                    continue
                obj2_spec = mod[2]
                # Try to find obj2 instance
                obj2 = obj2_spec
                obj2_base = get_base_object_name(obj2_spec)
                if obj2_spec not in all_objects_and_regions:
                    for obj in all_objects_and_regions:
                        if get_base_object_name(obj) == obj2_base:
                            obj2 = obj
                            break
                    else:
                        obj2 = obj2_spec
                predicate = f"(In {obj_name} {obj2})"
            elif predicate_type == 'Turnon':
                # Format: (Turnon obj_name)
                predicate = f"(Turnon {obj_name})"
            else:
                print(f"WARNING: Unknown predicate type: {predicate_type}")
                continue
            
            # Check if predicate already exists in init section
            # Escape parentheses in predicate for regex
            escaped_predicate = re.escape(predicate)
            if re.search(escaped_predicate, init_content):
                # Predicate already exists, skip
                continue
            
            # Find the insertion point (before the closing parenthesis of :init)
            # Insert before the last ')' of the init block, ensuring proper formatting
            # The closing parenthesis should be on its own line with 2 spaces indentation
            init_content_stripped = init_content.rstrip()
            if init_content_stripped.endswith(')'):
                # Find the last newline before the closing parenthesis
                # This ensures we insert before the closing parenthesis line
                last_newline_idx = init_content_stripped.rfind('\n')
                if last_newline_idx >= 0:
                    # Check if the line before the closing parenthesis is empty or has content
                    line_before_closing = init_content_stripped[last_newline_idx+1:].strip()
                    if line_before_closing == ')':
                        # Closing parenthesis is on its own line, insert before it
                        # Remove any trailing blank lines first
                        content_before_closing = init_content_stripped[:last_newline_idx].rstrip()
                        insertion_point = len(content_before_closing)
                        # Add newline, 4 spaces indentation for predicate
                        new_predicate = f"\n    {predicate}"
                        init_content = init_content[:insertion_point] + new_predicate + init_content[insertion_point:]
                    else:
                        # Last predicate is on the same line as closing parenthesis
                        # Insert before the closing parenthesis, add newline for closing parenthesis
                        insertion_point = len(init_content_stripped) - 1
                        new_predicate = f"\n    {predicate}\n  )"
                        init_content = init_content[:insertion_point] + new_predicate
                else:
                    # No newline found, insert before closing parenthesis
                    insertion_point = len(init_content_stripped) - 1
                    new_predicate = f"\n    {predicate}\n  )"
                    init_content = init_content[:insertion_point] + new_predicate
            else:
                # No closing parenthesis found, append at end with proper formatting
                init_content = init_content.rstrip() + f"\n    {predicate}\n  )"
        
        # Reconstruct the full BDDL with modified init section
        return text[:start] + init_content + text[end:]

    def perturb(self) -> str:
        new_lang = self.new_lang
        new_goal = self.new_goal
        new_objs = self.new_objs_of_interest

        new_content = self.parser.file_content
        if new_lang is not None:
            new_content = self._replace_language(new_content, new_lang)
        if new_goal is not None:
            new_content = self._replace_goal(new_content, new_goal)
        if new_objs is not None:
            new_content = self._replace_obj_of_interest(new_content, new_objs)
        return new_content


def infer_execution_steps_from_goal(
    goal_conditions: List[Dict[str, str]],
    objects_config: Dict,
    operators_config: Dict
) -> List[str]:
    """
    Infer execution steps from goal conditions parsed from a BDDL file.
    
    Args:
        goal_conditions: List of goal predicates from BDDLParser._parse_goal_conditions()
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
    
    Returns:
        List of execution steps (e.g., ["Grasp obj1", "Place obj1 obj2"])
    """
    execution_steps = []
    
    for goal_pred in goal_conditions:
        pred_type = goal_pred['type']
        
        if pred_type == 'Open':
            obj = goal_pred['obj1']
            execution_steps.append(f"Open {obj}")
            
        elif pred_type == 'Turnon':
            obj = goal_pred['obj1']
            execution_steps.append(f"Turnon {obj}")
            
        elif pred_type == 'In':
            container = goal_pred['obj1']  # container
            obj_to_place = goal_pred['obj2']  # object to place
            execution_steps.append(f"Open {container}")
            execution_steps.append(f"Grasp {obj_to_place}")
            execution_steps.append(f"Place {obj_to_place} {container}")
            
        elif pred_type == 'On':
            obj1 = goal_pred['obj1']  # object being moved
            obj2 = goal_pred['obj2']  # target location
            
            # Determine if this is a push or grasp variant
            # Check if obj1 has can_push operator and obj2 is in allowed_targets
            # Use get_object_config to match by base name (obj1 may be an instance like plate_1)
            is_push_variant = False
            obj1_config = get_object_config(obj1, objects_config)
            if obj1_config is not None:
                obj1_operators = obj1_config.get('operators', {})
                if isinstance(obj1_operators, dict) and 'can_push' in obj1_operators:
                    can_push_config = obj1_operators['can_push']
                    if isinstance(can_push_config, dict):
                        allowed_targets = can_push_config.get('allowed_targets')
                        if allowed_targets is not None:
                            # Check if obj2 matches any target in allowed_targets (by exact name or base name)
                            obj2_base = get_base_object_name(obj2)
                            if obj2 in allowed_targets or obj2_base in allowed_targets:
                                is_push_variant = True
                            else:
                                # Also check if any base name in allowed_targets matches obj2_base
                                allowed_targets_bases = [get_base_object_name(t) for t in allowed_targets]
                                if obj2_base in allowed_targets_bases:
                                    is_push_variant = True
            
            if is_push_variant:
                execution_steps.append(f"Touch {obj1}")
                execution_steps.append(f"Place {obj1} {obj2}")
            else:
                execution_steps.append(f"Grasp {obj1}")
                execution_steps.append(f"Place {obj1} {obj2}")
        else:
            raise ValueError(f"Unknown goal predicate type: {pred_type}")
    
    return execution_steps


def parse_execution_step(step_str: str) -> Tuple[str, str]:
    """Parse an execution step string like 'Grasp akita_black_bowl_1' into (stage, object)"""
    parts = step_str.strip().split()
    if len(parts) >= 2:
        return (parts[0], parts[1])
    elif len(parts) == 1:
        return (parts[0], "")
    return ("", "")


def extract_goal_relevant_entities(
    goal_conditions: Optional[List[Dict[str, str]]]
) -> Tuple[Set[str], Set[str]]:
    """
    Extract relevant object instances and target locations from goal conditions only.
    
    Args:
        goal_conditions: Optional list of goal predicates from BDDLParser
    
    Returns:
        Tuple of (relevant_objects, relevant_locations)
    """
    relevant_objects = set()
    relevant_locations = set()
    
    if goal_conditions:
        for goal_pred in goal_conditions:
            pred_type = goal_pred.get('type')
            obj1 = goal_pred.get('obj1')
            obj2 = goal_pred.get('obj2')
            if pred_type == 'On':
                if obj1:
                    relevant_objects.add(obj1)
                if obj2:
                    relevant_locations.add(obj2)
            elif pred_type == 'In':
                # container is obj1, object placed is obj2
                if obj2:
                    relevant_objects.add(obj2)
                if obj1:
                    relevant_locations.add(obj1)
            elif pred_type in ('Open', 'Turnon'):
                if obj1:
                    relevant_objects.add(obj1)
            else:
                # Fallback: include any provided entities
                if obj1:
                    relevant_objects.add(obj1)
                if obj2:
                    relevant_locations.add(obj2)
    
    return relevant_objects, relevant_locations


def get_full_location_chain(obj_name: str, init_states: Dict[str, str], visited: Optional[set] = None) -> Tuple[str, ...]:
    """
    Get the full location chain for an object, following the hierarchy recursively.
    
    For example, if bowl is on plate, and plate is on cabinet, returns:
    ('bowl', 'plate', 'cabinet')
    
    Args:
        obj_name: Name of the object
        init_states: Dictionary mapping object -> direct location
        visited: Set of objects already visited (to detect cycles)
        
    Returns:
        Tuple of object names in the location chain
    """
    if visited is None:
        visited = set()
    
    if obj_name in visited:
        # Cycle detected, return just this object
        return (obj_name,)
    
    visited.add(obj_name)
    
    # Get direct location
    direct_location = init_states.get(obj_name)
    if direct_location is None:
        # Object has no location, return just this object
        return (obj_name,)
    
    # Check if the location is itself an object that has a location
    if direct_location in init_states:
        # Recursively get the chain for the location
        location_chain = get_full_location_chain(direct_location, init_states, visited.copy())
        return (obj_name,) + location_chain  # Skip the first element since we already have obj_name
    else:
        # Location is a region/fixture, end of chain
        return (obj_name, direct_location)


def only_relevant_entities_and_locations_differ(
    existing_bddl_content: str,
    new_bddl_content: str,
    execution_steps: List[str]
) -> bool:
    """
    Check if only non-relevant entities differ in initial states.
    If only non-relevant entities differ, return True (should skip duplicate).
    If goal-relevant entities or locations differ, return False (should allow duplicate).
    
    This function checks the FULL location chain (e.g., bowl->plate->cabinet vs bowl->plate->stove)
    to catch cases where objects are on the same direct parent but that parent is on different surfaces.
    
    Args:
        existing_bddl_content: BDDL content of existing task
        new_bddl_content: BDDL content of new task
        execution_steps: Execution steps for the task
    
    Returns:
        True if only non-relevant entities differ (should skip), 
        False if goal-relevant entities or locations differ (should allow duplicate)
    """
    # Parse both BDDLs to get initial states and goals
    parser_existing = BDDLParser(existing_bddl_content)
    parser_new = BDDLParser(new_bddl_content)
    
    existing_init_states = parser_existing.initial_states
    new_init_states = parser_new.initial_states
    existing_goal_conditions = parser_existing.goal_conditions
    new_goal_conditions = parser_new.goal_conditions
    
    # Extract goal-relevant entities for both tasks
    existing_objects, existing_locations = extract_goal_relevant_entities(existing_goal_conditions)
    new_objects, new_locations = extract_goal_relevant_entities(new_goal_conditions)
    
    # Compare goal-relevant object location chains (base-normalized, order-insensitive)
    def build_goal_chain_signatures(relevant_objects: Set[str], init_states: Dict[str, str]) -> List[Tuple[str, ...]]:
        signatures = []
        for obj in relevant_objects:
            chain = get_full_location_chain(obj, init_states)
            signatures.append(tuple(get_base_object_name(item) for item in chain))
        return sorted(signatures)
    
    existing_chain_signatures = build_goal_chain_signatures(existing_objects, existing_init_states)
    new_chain_signatures = build_goal_chain_signatures(new_objects, new_init_states)
    if existing_chain_signatures != new_chain_signatures:
        return False
    
    # If goal-relevant target locations differ, allow duplicate
    relevant_locations_existing = sorted(get_base_object_name(loc) for loc in existing_locations)
    relevant_locations_new = sorted(get_base_object_name(loc) for loc in new_locations)
    if relevant_locations_existing != relevant_locations_new:
        return False
    
    # All goal-relevant objects have the same initial state chain
    # Check if any non-relevant object differs
    relevant_objects = existing_objects | new_objects
    all_objects = set(existing_init_states.keys()) | set(new_init_states.keys())
    for obj in all_objects:
        if obj not in relevant_objects:
            existing_chain = get_full_location_chain(obj, existing_init_states)
            new_chain = get_full_location_chain(obj, new_init_states)
            
            # If a non-relevant object differs, this is the only difference
            if existing_chain != new_chain:
                return True  # Only non-interacted objects differ, skip duplicate
    
    # No differences at all (shouldn't happen if we're checking for duplicates)
    return True


def find_actions_from(
    existing_tasks: Dict[str, Dict], 
    execution_steps: List[str],
    default_actions_from_strategy: Optional[Dict[str, str]] = None,
    initial_states: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Find the task to use for actions_from.
    
    Priority:
    1. default_actions_from_strategy - default task for this first step
    2. First existing task (alphabetically) that matches the first execution step
    
    Matches require:
    - Exact object instances in execution steps (not just base names)
    - Same init states for all objects (all objects must be in the same locations)
    
    This ensures action replay will work correctly since the exact same objects
    start in the exact same positions.
    
    Args:
        existing_tasks: Dictionary of existing task metadata
        execution_steps: Execution steps for the new task
        default_actions_from_strategy: Optional default strategy mapping first step to task
        initial_states: Full init state dict (object -> location) for the new task
    """
    if not execution_steps:
        return None
    
    first_step = execution_steps[0]
    
    # Check if there's a default strategy for this first step
    if default_actions_from_strategy and first_step in default_actions_from_strategy:
        return default_actions_from_strategy[first_step]
    
    stage, obj = parse_execution_step(first_step)
    
    # Find all candidates whose first step matches, then pick the one with the longest
    # matching prefix of execution steps (tie-broken alphabetically).
    sorted_tasks = sorted(existing_tasks.items())

    best_task_name = None
    best_match_len = 0

    for task_name, task_metadata in sorted_tasks:
        task_steps = task_metadata.get('execution_steps', [])
        if not task_steps:
            continue

        task_first_step = task_steps[0]
        task_stage, task_obj = parse_execution_step(task_first_step)

        # Match if same stage and exact same object instance (not base name)
        if task_stage != stage or task_obj != obj:
            continue

        # Require full init state matching: all objects must be in the same locations
        if initial_states is not None:
            task_initial_states = task_metadata.get('initial_states')
            if task_initial_states is None:
                # Existing task doesn't have init state info, skip it
                continue

            # Compare init states: must match exactly (same objects in same locations)
            if task_initial_states != initial_states:
                continue

        # Count how many leading execution steps match
        match_len = 0
        for new_step, task_step in zip(execution_steps, task_steps):
            if new_step == task_step:
                match_len += 1
            else:
                break

        # Prefer longer prefix match; alphabetical order (already sorted) breaks ties
        if match_len > best_match_len:
            best_match_len = match_len
            best_task_name = task_name

    return best_task_name


def find_grasps_from(
    existing_tasks: Dict[str, Dict],
    object_name: str,
    task_override: Optional[str] = None,
    default_grasp_strategy: Optional[Dict[str, str]] = None,
    object_location: Optional[str] = None,
    place_location: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Find the task and object to use for grasp strategy.

    Returns:
        Tuple of (task_name, object_name) from the source task, or None if no match found.
        The object_name is the actual object name in the source task (e.g., "bowl_2" even
        if we're looking for "bowl_1").

    Priority:
    1. task_override - specific override for this task
    2. default_grasp_strategy - default task for this object (matches by base name)
    3. First existing task (alphabetically) that grasps this object at the same location
       AND places it at the same place_location (by base name)
    4. First existing task (alphabetically) that grasps this object at the same location

    Matching criteria:
    - Matches by base object name (ignoring instance numbers) to handle cases where
      new tasks use different object instances (e.g., akita_black_bowl_2) than existing
      tasks (e.g., akita_black_bowl_1).
    - If object_location is provided, the existing task's grasped object must be at
      the exact same location (e.g., cabinet_1 must match cabinet_1 exactly).
      This ensures we only reuse grasps from objects in the same location.
    - If place_location is provided, tasks that also place the object at the same target
      location (by base name) are preferred over those that don't.

    Args:
        existing_tasks: Dictionary of existing task metadata
        object_name: Name of the object to find a grasp for (e.g., "bowl_2")
        task_override: Optional specific task override (task name only, object will be found)
        default_grasp_strategy: Optional default strategy mapping object to task
        object_location: Optional location of the object (e.g., "cabinet_1"). If provided,
                         only matches tasks where the grasped object is at this exact location.
        place_location: Optional target place location in the current task (e.g.,
                        "wooden_cabinet_1_top_side"). If provided, tasks that place the object
                        at the same location (by base name) are preferred.
    """
    if task_override:
        # If task_override is provided, find the grasped object in that task
        override_metadata = existing_tasks.get(task_override)
        if override_metadata:
            override_steps = override_metadata.get('execution_steps', [])
            object_name_base = get_base_object_name(object_name)
            for step in override_steps:
                stage, obj = parse_execution_step(step)
                if stage == 'Grasp' and obj:
                    obj_base = get_base_object_name(obj)
                    if obj_base == object_name_base:
                        # If location is required, verify it matches
                        if object_location is not None:
                            override_initial_states = override_metadata.get('initial_states')
                            if override_initial_states:
                                obj_loc = override_initial_states.get(obj)
                                if obj_loc == object_location:
                                    return (task_override, obj)
                        else:
                            return (task_override, obj)
        # If we can't find a matching object in the override task, return None
        return None
    
    # For default_grasp_strategy, match by base name if exact match not found
    object_name_base = get_base_object_name(object_name)
    if default_grasp_strategy:
        # Helper function to find the grasped object in a candidate task
        def find_grasped_object_in_task(candidate_task: str) -> Optional[str]:
            """Find the grasped object in candidate task that matches our criteria."""
            candidate_metadata = existing_tasks.get(candidate_task)
            if not candidate_metadata:
                return None
            candidate_initial_states = candidate_metadata.get('initial_states')
            if not candidate_initial_states:
                # If no initial states, only allow if no location requirement
                if object_location is not None:
                    return None
            # Find the grasped object in the candidate task
            candidate_steps = candidate_metadata.get('execution_steps', [])
            for step in candidate_steps:
                stage, obj = parse_execution_step(step)
                if stage == 'Grasp' and obj:
                    # Verify the grasped object has the same base name
                    obj_base = get_base_object_name(obj)
                    if obj_base != object_name_base:
                        continue
                    # If location is required, verify it matches
                    if object_location is not None:
                        if candidate_initial_states:
                            candidate_obj_location = candidate_initial_states.get(obj)
                            if candidate_obj_location == object_location:
                                return obj
                    else:
                        # No location requirement, base name matches
                        return obj
            return None
        
        # First try exact match
        if object_name in default_grasp_strategy:
            candidate_task = default_grasp_strategy[object_name]
            grasped_obj = find_grasped_object_in_task(candidate_task)
            if grasped_obj is not None:
                return (candidate_task, grasped_obj)
        # Then try base name match
        if object_name_base in default_grasp_strategy:
            candidate_task = default_grasp_strategy[object_name_base]
            grasped_obj = find_grasped_object_in_task(candidate_task)
            if grasped_obj is not None:
                return (candidate_task, grasped_obj)
        # Also check if any key in default_grasp_strategy has the same base name
        for key, value in default_grasp_strategy.items():
            if get_base_object_name(key) == object_name_base:
                candidate_task = value
                grasped_obj = find_grasped_object_in_task(candidate_task)
                if grasped_obj is not None:
                    return (candidate_task, grasped_obj)
    
    # Sort existing tasks alphabetically
    sorted_tasks = sorted(existing_tasks.items())
    place_location_base = get_base_object_name(place_location) if place_location else None

    def _find_in_sorted_tasks(require_place_location_match: bool) -> Optional[Tuple[str, str]]:
        for task_name, task_metadata in sorted_tasks:
            task_steps = task_metadata.get('execution_steps', [])
            task_initial_states = task_metadata.get('initial_states')

            for i, step in enumerate(task_steps):
                stage, obj = parse_execution_step(step)
                if stage != 'Grasp' or not obj:
                    continue
                obj_base = get_base_object_name(obj)
                if obj_base != object_name_base:
                    continue
                # If object_location is provided, verify the grasped object is at the same location
                if object_location is not None:
                    if task_initial_states is None:
                        continue
                    obj_location = task_initial_states.get(obj)
                    if obj_location != object_location:
                        continue
                # If require_place_location_match, check that the following Place step
                # targets the same location (by base name)
                if require_place_location_match and place_location_base is not None:
                    place_step_matched = False
                    for j in range(i + 1, len(task_steps)):
                        place_stage, _ = parse_execution_step(task_steps[j])
                        if place_stage != 'Place':
                            break
                        place_parts = task_steps[j].split()
                        if len(place_parts) >= 3:
                            task_place_location_base = get_base_object_name(place_parts[2])
                            if task_place_location_base == place_location_base:
                                place_step_matched = True
                        break
                    if not place_step_matched:
                        continue
                return (task_name, obj)
        return None

    # Priority 3: match same place location
    if place_location_base is not None:
        result = _find_in_sorted_tasks(require_place_location_match=True)
        if result is not None:
            return result

    # Priority 4: any alphabetically first match
    return _find_in_sorted_tasks(require_place_location_match=False)

def to_sentence_case(text: str) -> str:
    """Normalize text so only the first character is uppercase."""
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:].lower()


def apply_object_overrides(objects_config: Dict, object_overrides: Dict) -> Dict:
    """
    Apply object overrides to create a split-specific objects_config.
    
    Args:
        objects_config: Base objects configuration from YAML
        object_overrides: Overrides from split config (e.g., {'grasp_objects': [...]})
    
    Returns:
        Modified objects_config with overrides applied
    """
    # Create a deep copy to avoid modifying the original
    split_objects_config = copy.deepcopy(objects_config)
    
    def remove_operator(obj_config: Dict, operator_name: str):
        """Helper function to remove an operator from an object's operators."""
        operators = obj_config.get('operators', {})
        if isinstance(operators, dict):
            operators.pop(operator_name, None)
        elif isinstance(operators, list):
            # Handle old list format
            if operator_name in operators:
                operators.remove(operator_name)
    
    def matches_object_list(obj_name: str, object_list: set) -> bool:
        """Check if obj_name matches any entry in object_list (by exact match or base name)."""
        if obj_name in object_list:
            return True
        obj_base = get_base_object_name(obj_name)
        if obj_base in object_list:
            return True
        # Check if any entry in object_list has the same base name as obj_name
        for list_entry in object_list:
            if get_base_object_name(list_entry) == obj_base:
                return True
        return False
    
    # Handle grasp_objects: remove can_grasp from objects not in the list
    if 'grasp_objects' in object_overrides:
        grasp_objects = set(object_overrides['grasp_objects'])
        
        # Remove can_grasp operator from objects not in grasp_objects list
        for obj_name, obj_config in split_objects_config.items():
            if not matches_object_list(obj_name, grasp_objects):
                remove_operator(obj_config, 'can_grasp')
    
    # Handle open_objects: remove can_open from objects not in the list
    if 'open_objects' in object_overrides:
        open_objects = set(object_overrides['open_objects'])
        
        # Remove can_open operator from objects not in open_objects list
        for obj_name, obj_config in split_objects_config.items():
            if not matches_object_list(obj_name, open_objects):
                remove_operator(obj_config, 'can_open')
    
    # Handle push_objects: remove can_push from objects not in the list
    if 'push_objects' in object_overrides:
        push_objects = set(object_overrides['push_objects'])
        
        # Remove can_push operator from objects not in push_objects list
        for obj_name, obj_config in split_objects_config.items():
            if not matches_object_list(obj_name, push_objects):
                remove_operator(obj_config, 'can_push')
    
    # Handle turn_on_objects: remove can_turn_on from objects not in the list
    if 'turn_on_objects' in object_overrides:
        turn_on_objects = set(object_overrides['turn_on_objects'])
        
        # Remove can_turn_on operator from objects not in turn_on_objects list
        for obj_name, obj_config in split_objects_config.items():
            if not matches_object_list(obj_name, turn_on_objects):
                remove_operator(obj_config, 'can_turn_on')
    
    # Handle object_property_overrides: override specific properties for objects
    # Format: {object_name: {property_name: value}}
    # Supports nested operators like can_place_on, can_grasp, etc.
    # obj_name may be a base name or specific instance - match to split_objects_config
    if 'object_property_overrides' in object_overrides:
        property_overrides = object_overrides['object_property_overrides']
        for obj_name, property_updates in property_overrides.items():
            # Match obj_name to a key in split_objects_config (by exact match or base name)
            config_key = match_object_to_config(obj_name, split_objects_config)
            if config_key:
                obj_config = split_objects_config[config_key]
                
                # Handle nested operator properties (e.g., can_place_on, can_grasp)
                # These need to be merged into the operators dict rather than replacing it
                for prop_name, prop_value in property_updates.items():
                    if prop_name in ['can_place_on', 'can_grasp', 'can_push', 'can_open', 'can_turn_on']:
                        # This is an operator property - merge it into operators
                        if 'operators' not in obj_config:
                            obj_config['operators'] = {}
                        if not isinstance(obj_config['operators'], dict):
                            obj_config['operators'] = {}
                        
                        # Merge the operator config (prop_value is a dict with operator properties)
                        if isinstance(prop_value, dict):
                            if prop_name not in obj_config['operators']:
                                obj_config['operators'][prop_name] = {}
                            obj_config['operators'][prop_name].update(prop_value)
                        else:
                            obj_config['operators'][prop_name] = prop_value
                    else:
                        # Regular property - update directly
                        obj_config[prop_name] = prop_value
    
    return split_objects_config


def get_base_object_name(obj_name: str) -> str:
    """
    Extract the base object name by removing numeric instance suffix (_1, _2, etc.).
    Handles cases where the suffix appears at the end or in the middle of the name.
    
    Examples:
        akita_black_bowl_1 -> akita_black_bowl
        plate_2 -> plate
        flat_stove_1 -> flat_stove
        flat_stove_1_cook_region -> flat_stove_cook_region
        wine_rack_1_top_region -> wine_rack_top_region
        wooden_cabinet_1_middle_region -> wooden_cabinet_middle_region
        wooden_cabinet_1_top_side -> wooden_cabinet_top_side
        main_table_stove_front_region -> main_table_stove_front_region (no change if no numeric suffix)
    """
    # Split by underscores and filter out parts that are purely numeric (instance numbers)
    parts = obj_name.split('_')
    filtered_parts = []
    for part in parts:
        # Skip parts that are purely numeric (these are instance numbers like "1", "2")
        if not part.isdigit():
            filtered_parts.append(part)
    
    # Rejoin with underscores
    if filtered_parts:
        return '_'.join(filtered_parts)
    
    # Edge case: if all parts were numeric, return original (shouldn't happen in practice)
    return obj_name


def build_base_to_instances_mapping(objects_in_bddl: List[str]) -> Dict[str, List[str]]:
    """
    Build a mapping from base object names to all instances found in BDDL files.
    
    Returns:
        Dict mapping base_name -> [instance1, instance2, ...]
    """
    base_to_instances = {}
    for obj_name in objects_in_bddl:
        base_name = get_base_object_name(obj_name)
        if base_name not in base_to_instances:
            base_to_instances[base_name] = []
        base_to_instances[base_name].append(obj_name)
    return base_to_instances


def match_object_to_config(obj_name: str, objects_config: Dict) -> Optional[str]:
    """
    Match an object name (which may be a specific instance like akita_black_bowl_1)
    to a key in objects_config (which may be a base name like akita_black_bowl).
    
    Returns:
        The matching key from objects_config, or None if no match found.
    """
    # First try exact match
    if obj_name in objects_config:
        return obj_name
    
    # Try base name match
    base_name = get_base_object_name(obj_name)
    if base_name in objects_config:
        return base_name
    
    return None


def get_object_config(obj_name: str, objects_config: Dict) -> Optional[Dict]:
    """
    Get the configuration for an object, matching by base name if needed.
    
    Returns:
        The object configuration dict, or None if not found.
    """
    config_key = match_object_to_config(obj_name, objects_config)
    if config_key:
        return objects_config[config_key]
    return None


def generate_operator_combinations(
    objects_in_bddl: List[str],
    objects_config: Dict,
    operators_config: Dict
) -> List[Dict]:
    """
    Generate all valid operator combinations from objects in the BDDL.
    Returns a list of task specifications.
    Only considers objects that match (by base name) objects in objects_config.
    Generates tasks for all instances that match a base object name.
    """
    tasks = []
    
    # Build mapping from base names to all instances
    base_to_instances = build_base_to_instances_mapping(objects_in_bddl)
    
    # Build a mapping of object name to its operators
    # Match objects by base name to objects_config
    obj_to_operators = {}
    for obj_name in objects_in_bddl:
        config_key = match_object_to_config(obj_name, objects_config)
        if config_key:
            # Operators can be a list (old format) or dict (new format)
            ops = objects_config[config_key].get('operators', [])
            if isinstance(ops, list):
                obj_to_operators[obj_name] = set(ops)
            else:
                # It's a dict, extract the keys
                obj_to_operators[obj_name] = set(ops.keys())
    
    # Generate On tasks (grasp variant)
    # Only targets with can_place_on (not can_place_in, which requires opening first)
    for obj1_name, obj1_ops in obj_to_operators.items():
        if 'can_grasp' in obj1_ops:
            # Get config for obj1 (may be base name)
            obj1_config = get_object_config(obj1_name, objects_config)
            if obj1_config is None:
                continue
            obj1_operators = obj1_config.get('operators', {})
            obj1_can_place_on_config = obj1_operators.get('can_place_on', {})
            allowed_targets = obj1_can_place_on_config.get('allowed_targets') if isinstance(obj1_can_place_on_config, dict) else None
            
            for obj2_name, obj2_ops in obj_to_operators.items():
                if obj1_name == obj2_name:
                    continue
                if 'can_place_on' not in obj2_ops:
                    continue
                # If allowed_targets is specified on the source object, check if obj2 matches
                # (allowed_targets may contain base names or specific instances)
                if allowed_targets is not None:
                    obj2_base = get_base_object_name(obj2_name)
                    # Check if obj2_name or obj2_base is in allowed_targets
                    if obj2_name not in allowed_targets and obj2_base not in allowed_targets:
                        # Also check if any base name in allowed_targets matches obj2_base
                        allowed_targets_bases = [get_base_object_name(t) for t in allowed_targets]
                        if obj2_base not in allowed_targets_bases:
                            continue
                # Check for allowed_sources restriction on the target's can_place_on operator
                obj2_config = get_object_config(obj2_name, objects_config)
                if obj2_config is None:
                    continue
                obj2_operators = obj2_config.get('operators', {})
                can_place_on_config = obj2_operators.get('can_place_on', {})
                allowed_sources = can_place_on_config.get('allowed_sources') if isinstance(can_place_on_config, dict) else None
                if allowed_sources is not None:
                    obj1_base = get_base_object_name(obj1_name)
                    # Check if obj1_name or obj1_base is in allowed_sources
                    if obj1_name not in allowed_sources and obj1_base not in allowed_sources:
                        # Also check if any base name in allowed_sources matches obj1_base
                        allowed_sources_bases = [get_base_object_name(s) for s in allowed_sources]
                        if obj1_base not in allowed_sources_bases:
                            continue
                tasks.append({
                    'type': 'On',
                    'variant': 'grasp',
                    'obj1': obj1_name,
                    'obj2': obj2_name,
                })
    
    # Generate On tasks (push variant)
    # Only targets with can_place_on (not can_place_in, which requires opening first)
    for obj1_name, obj1_ops in obj_to_operators.items():
        if 'can_push' in obj1_ops:
            # Get config for obj1 (may be base name)
            obj1_config = get_object_config(obj1_name, objects_config)
            if obj1_config is None:
                continue
            obj1_operators = obj1_config.get('operators', {})
            can_push_config = obj1_operators.get('can_push', {})
            allowed_targets = can_push_config.get('allowed_targets') if isinstance(can_push_config, dict) else None
            
            for obj2_name, obj2_ops in obj_to_operators.items():
                if obj1_name == obj2_name:
                    continue
                if 'can_place_on' not in obj2_ops:
                    continue
                # If allowed_targets is specified, check if obj2 matches (by name or base name)
                if allowed_targets is not None:
                    obj2_base = get_base_object_name(obj2_name)
                    if obj2_name not in allowed_targets and obj2_base not in allowed_targets:
                        allowed_targets_bases = [get_base_object_name(t) for t in allowed_targets]
                        if obj2_base not in allowed_targets_bases:
                            continue
                # Check for allowed_sources restriction on the target's can_place_on operator
                obj2_config = get_object_config(obj2_name, objects_config)
                if obj2_config is None:
                    continue
                obj2_operators = obj2_config.get('operators', {})
                can_place_on_config = obj2_operators.get('can_place_on', {})
                allowed_sources = can_place_on_config.get('allowed_sources') if isinstance(can_place_on_config, dict) else None
                if allowed_sources is not None:
                    obj1_base = get_base_object_name(obj1_name)
                    if obj1_name not in allowed_sources and obj1_base not in allowed_sources:
                        allowed_sources_bases = [get_base_object_name(s) for s in allowed_sources]
                        if obj1_base not in allowed_sources_bases:
                            continue
                tasks.append({
                    'type': 'On',
                    'variant': 'push',
                    'obj1': obj1_name,
                    'obj2': obj2_name,
                })
    
    # Generate In tasks (container must be opened first, then object placed inside)
    for obj1_name, obj1_ops in obj_to_operators.items():
        if 'can_open' in obj1_ops and 'can_place_in' in obj1_ops:
            for obj2_name, obj2_ops in obj_to_operators.items():
                if obj1_name == obj2_name:
                    continue
                if 'can_grasp' in obj2_ops:
                    tasks.append({
                        'type': 'In',
                        'obj1': obj1_name,  # container (drawer)
                        'obj2': obj2_name,  # object to place inside
                    })

    # Generate PlaceIn tasks (container already open; valid only as 2nd action after Open)
    # is_action_valid enforces that the container must already be in world_state.opened_objects,
    # so PlaceIn can never be a valid first action and will only appear as a second action.
    for obj1_name, obj1_ops in obj_to_operators.items():
        if 'can_place_in' in obj1_ops:
            for obj2_name, obj2_ops in obj_to_operators.items():
                if obj1_name == obj2_name:
                    continue
                if 'can_grasp' in obj2_ops:
                    tasks.append({
                        'type': 'PlaceIn',
                        'obj1': obj1_name,  # container (already-open drawer)
                        'obj2': obj2_name,  # object to place inside
                    })
    
    # Generate Open tasks
    for obj_name, obj_ops in obj_to_operators.items():
        if 'can_open' in obj_ops:
            tasks.append({
                'type': 'Open',
                'obj1': obj_name,
            })
    
    # Generate Turnon tasks
    for obj_name, obj_ops in obj_to_operators.items():
        if 'can_turn_on' in obj_ops:
            tasks.append({
                'type': 'Turnon',
                'obj1': obj_name,
            })
    
    return tasks


def get_task_object_operator_pairs(task_spec: Dict) -> List[Tuple[str, str]]:
    """
    Get all (object, operator) pairs involved in a task specification.
    Returns list of tuples (object_name, operator_name).
    """
    pairs = []
    task_type = task_spec['type']
    
    if task_type == 'On':
        variant = task_spec.get('variant', 'grasp')
        if variant == 'grasp':
            pairs.append((task_spec['obj1'], 'can_grasp'))
            pairs.append((task_spec['obj2'], 'can_place_on'))
        else:  # push
            pairs.append((task_spec['obj1'], 'can_push'))
            pairs.append((task_spec['obj2'], 'can_place_on'))
    elif task_type == 'In':
        pairs.append((task_spec['obj1'], 'can_open'))
        pairs.append((task_spec['obj1'], 'can_place_in'))
        pairs.append((task_spec['obj2'], 'can_grasp'))
    elif task_type == 'PlaceIn':
        pairs.append((task_spec['obj1'], 'can_place_in'))
        pairs.append((task_spec['obj2'], 'can_grasp'))
    elif task_type == 'Open':
        pairs.append((task_spec['obj1'], 'can_open'))
    elif task_type == 'Turnon':
        pairs.append((task_spec['obj1'], 'can_turn_on'))

    return pairs


def target_matches(target_spec, action_target_base: str, action_target_instance: str, initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a target specification matches an action target.
    
    Args:
        target_spec: Can be:
            - A string: matches if action_target_base equals it
            - A list of strings: matches if any string in list equals action_target_base
            - A list containing lists: each inner list [object, initial_location] matches if:
                - object matches action_target_base (by base name)
                - initial_location matches the initial location of action_target_instance (by base name)
        action_target_base: Base name of the target object from the action (e.g., 'cookies')
        action_target_instance: Instance name of the target object from the action (e.g., 'cookies_1')
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if target_spec matches the action target
    """
    if target_spec is None:
        return False
    
    # Strip _resized suffix from action_target_base for comparison (resized regions are functionally equivalent)
    action_target_base_stripped = _strip_resized_suffix(action_target_base)
    
    # If target_spec is a string, check direct match
    if isinstance(target_spec, str):
        return action_target_base_stripped == target_spec
    
    # If target_spec is a list, check each entry
    if isinstance(target_spec, list):
        for entry in target_spec:
            if isinstance(entry, str):
                # Simple string match
                if action_target_base_stripped == entry:
                    return True
            elif isinstance(entry, list) and len(entry) == 2:
                # List entry with [object, initial_location]
                required_object = entry[0]
                required_initial_location = entry[1]
                
                # Check if object matches (by base name, stripping _resized)
                if action_target_base_stripped != required_object:
                    continue
                
                # Check if initial location matches
                if initial_states is None:
                    continue
                
                # Get the initial location of the target object instance
                target_initial_location = initial_states.get(action_target_instance)
                if target_initial_location is None:
                    continue
                
                # Compare by base name (strip _resized suffix)
                target_initial_location_base = get_base_object_name(target_initial_location)
                target_initial_location_base_stripped = _strip_resized_suffix(target_initial_location_base)
                required_initial_location_base = get_base_object_name(required_initial_location)
                if target_initial_location_base_stripped == required_initial_location_base:
                    return True
    
    return False


def action_matches_operator_spec(action: Dict, operator_spec: Dict, initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a single action matches an operator specification.
    
    Args:
        action: A single action task spec with specific object instances (e.g., bowl_1)
        operator_spec: Dict with 'operator', 'source', 'target' (target optional for some ops),
                       and optionally 'initial_location' to check where source object starts
                       Uses general object names (e.g., bowl)
                       'target' can be a string, list of strings, or list containing [object, initial_location] pairs
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if the action matches the operator spec
    """
    op_type = operator_spec['operator']
    source = operator_spec.get('source')
    target = operator_spec.get('target')
    except_target = operator_spec.get('except_target')
    except_source = operator_spec.get('except_source')
    required_initial_location = operator_spec.get('initial_location')
    
    action_type = action['type']
    
    # Convert specific object instances in action to general names for comparison
    if op_type == 'can_place_on':
        # For On tasks with grasp variant, source is obj1, target is obj2
        if action_type == 'On' and action.get('variant', 'grasp') == 'grasp':
            obj1_base = get_base_object_name(action['obj1'])
            obj2_base = get_base_object_name(action['obj2'])
            if obj1_base == source:
                # Check except_source: if source object's initial location matches any except_source pattern, exclude it
                if except_source is not None:
                    if initial_states is None:
                        return False
                    obj1_initial_location = initial_states.get(action['obj1'])
                    if obj1_initial_location is not None:
                        obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                        except_source_base = get_base_object_name(except_source)
                        if obj1_initial_location_base == except_source_base:
                            # Source object is at an excluded initial location, don't match this operator spec
                            return False
                
                # Check if target matches (supports list of targets)
                # If target is None, treat as "match any target"
                if target is None or target_matches(target, obj2_base, action['obj2'], initial_states):
                    # Check except_target: if target matches any except_target pattern, exclude it
                    if except_target is not None:
                        if target_matches(except_target, obj2_base, action['obj2'], initial_states):
                            # Target matches an excluded pattern, don't match this operator spec
                            return False
                    
                    # Check initial location if specified
                    if required_initial_location is not None:
                        if initial_states is None:
                            return False
                        # Get the initial location of the source object (obj1)
                        obj1_initial_location = initial_states.get(action['obj1'])
                        if obj1_initial_location is None:
                            return False
                        # Compare by base name (initial_location may be a region with instance number)
                        obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                        required_initial_location_base = get_base_object_name(required_initial_location)
                        return obj1_initial_location_base == required_initial_location_base
                    return True
    elif op_type == 'can_place_in':
        # For In/PlaceIn tasks, source is obj2 (object to place), target is obj1 (container)
        if action_type in ('In', 'PlaceIn'):
            obj2_base = get_base_object_name(action['obj2'])
            obj1_base = get_base_object_name(action['obj1'])
            if obj2_base == source:
                # Check except_source: if source object's initial location matches any except_source pattern, exclude it
                if except_source is not None:
                    if initial_states is None:
                        return False
                    obj2_initial_location = initial_states.get(action['obj2'])
                    if obj2_initial_location is not None:
                        obj2_initial_location_base = get_base_object_name(obj2_initial_location)
                        except_source_base = get_base_object_name(except_source)
                        if obj2_initial_location_base == except_source_base:
                            # Source object is at an excluded initial location, don't match this operator spec
                            return False
                
                # Check if target matches (supports list of targets)
                # If target is None, treat as "match any target"
                if target is None or target_matches(target, obj1_base, action['obj1'], initial_states):
                    # Check except_target: if target matches any except_target pattern, exclude it
                    if except_target is not None:
                        if target_matches(except_target, obj1_base, action['obj1'], initial_states):
                            # Target matches an excluded pattern, don't match this operator spec
                            return False
                    
                    # Check initial location if specified
                    if required_initial_location is not None:
                        if initial_states is None:
                            return False
                        # Get the initial location of the source object (obj2)
                        obj2_initial_location = initial_states.get(action['obj2'])
                        if obj2_initial_location is None:
                            return False
                        # Compare by base name
                        obj2_initial_location_base = get_base_object_name(obj2_initial_location)
                        required_initial_location_base = get_base_object_name(required_initial_location)
                        return obj2_initial_location_base == required_initial_location_base
                    return True
    elif op_type == 'can_open':
        if action_type == 'Open':
            obj1_base = get_base_object_name(action['obj1'])
            # Check if target matches (supports list of targets)
            return target_matches(target, obj1_base, action['obj1'], initial_states)
    elif op_type == 'can_turn_on':
        if action_type == 'Turnon':
            obj1_base = get_base_object_name(action['obj1'])
            # Check if target matches (supports list of targets)
            return target_matches(target, obj1_base, action['obj1'], initial_states)
    elif op_type == 'can_push':
        if action_type == 'On' and action.get('variant') == 'push':
            obj1_base = get_base_object_name(action['obj1'])
            obj2_base = get_base_object_name(action['obj2'])
            if obj1_base == source:
                # Check except_source: if source object's initial location matches any except_source pattern, exclude it
                if except_source is not None:
                    if initial_states is None:
                        return False
                    obj1_initial_location = initial_states.get(action['obj1'])
                    if obj1_initial_location is not None:
                        obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                        except_source_base = get_base_object_name(except_source)
                        if obj1_initial_location_base == except_source_base:
                            # Source object is at an excluded initial location, don't match this operator spec
                            return False
                
                # Check if target matches (supports list of targets)
                # If target is None, treat as "match any target"
                if target is None or target_matches(target, obj2_base, action['obj2'], initial_states):
                    # Check except_target: if target matches any except_target pattern, exclude it
                    if except_target is not None:
                        if target_matches(except_target, obj2_base, action['obj2'], initial_states):
                            # Target matches an excluded pattern, don't match this operator spec
                            return False
                    
                    # Check initial location if specified
                    if required_initial_location is not None:
                        if initial_states is None:
                            return False
                        obj1_initial_location = initial_states.get(action['obj1'])
                        if obj1_initial_location is None:
                            return False
                        obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                        required_initial_location_base = get_base_object_name(required_initial_location)
                        return obj1_initial_location_base == required_initial_location_base
                    return True
    elif op_type == 'can_grasp':
        # Grasp is part of On/In tasks, check if obj1 (On) or obj2 (In) matches source
        if action_type == 'On' and action.get('variant', 'grasp') == 'grasp':
            obj1_base = get_base_object_name(action['obj1'])
            if obj1_base == source:
                # Check except_source: if source object's initial location matches any except_source pattern, exclude it
                if except_source is not None:
                    if initial_states is None:
                        return False
                    obj1_initial_location = initial_states.get(action['obj1'])
                    if obj1_initial_location is not None:
                        obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                        except_source_base = get_base_object_name(except_source)
                        if obj1_initial_location_base == except_source_base:
                            # Source object is at an excluded initial location, don't match this operator spec
                            return False
                
                # Check initial location if specified
                if required_initial_location is not None:
                    if initial_states is None:
                        return False
                    obj1_initial_location = initial_states.get(action['obj1'])
                    if obj1_initial_location is None:
                        return False
                    obj1_initial_location_base = get_base_object_name(obj1_initial_location)
                    required_initial_location_base = get_base_object_name(required_initial_location)
                    return obj1_initial_location_base == required_initial_location_base
                return True
        elif action_type in ('In', 'PlaceIn'):
            obj2_base = get_base_object_name(action['obj2'])
            if obj2_base == source:
                # Check except_source: if source object's initial location matches any except_source pattern, exclude it
                if except_source is not None:
                    if initial_states is None:
                        return False
                    obj2_initial_location = initial_states.get(action['obj2'])
                    if obj2_initial_location is not None:
                        obj2_initial_location_base = get_base_object_name(obj2_initial_location)
                        except_source_base = get_base_object_name(except_source)
                        if obj2_initial_location_base == except_source_base:
                            # Source object is at an excluded initial location, don't match this operator spec
                            return False

                # Check initial location if specified
                if required_initial_location is not None:
                    if initial_states is None:
                        return False
                    obj2_initial_location = initial_states.get(action['obj2'])
                    if obj2_initial_location is None:
                        return False
                    obj2_initial_location_base = get_base_object_name(obj2_initial_location)
                    required_initial_location_base = get_base_object_name(required_initial_location)
                    return obj2_initial_location_base == required_initial_location_base
                return True
    
    return False


def determine_on_variant(execution_steps: List[str]) -> str:
    """
    Determine if an On task is 'grasp' or 'push' variant based on execution steps.
    
    Args:
        execution_steps: List of execution step strings (e.g., ["Grasp obj1", "Place obj1 obj2"])
    
    Returns:
        'push' if first step is "Touch", 'grasp' otherwise
    """
    if not execution_steps:
        return 'grasp'  # Default to grasp
    first_step = execution_steps[0]
    if first_step.startswith("Touch"):
        return 'push'
    return 'grasp'


def _variant_for_on_action_at_step(execution_steps: List[str], step_idx: int) -> str:
    """
    Get variant ('push' or 'grasp') for an On action that starts at the given step index.
    """
    if not execution_steps or step_idx < 0 or step_idx >= len(execution_steps):
        return 'grasp'
    if execution_steps[step_idx].startswith("Touch"):
        return 'push'
    return 'grasp'


def _steps_consumed_by_goal_type(goal_type: str) -> int:
    """Number of execution steps consumed by this goal type (Open=1, Turnon=1, In=3, On=2)."""
    if goal_type in ('Open', 'Turnon'):
        return 1
    if goal_type == 'In':
        return 3
    if goal_type == 'On':
        return 2
    return 0


def task_matches_view_operators(goal_conditions: List[Dict], execution_steps: List[str], view_operators: List, initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a task matches any view operator specification.
    
    Supports both single operators and operator chains.
    Items in view_operators can be either:
      - dict: single operator spec
      - list: operator chain (list of operator specs in order)
    
    Converts goal conditions to action format (adding variant for On tasks) and checks
    if any action matches any view operator or if the task matches any operator chain.
    
    Args:
        goal_conditions: List of goal predicates from BDDLParser
        execution_steps: List of execution step strings
        view_operators: List of view operator specs/chains (can be dicts or lists)
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if any action in the task matches any view operator OR if task matches any chain
    """
    # If no operators are specified for this view, accept all tasks and
    # rely on other view-level filters (e.g., source_dirs, chain length).
    if not view_operators:
        return True

    # Convert goal conditions to actions format with per-action variant for On goals
    actions = []
    step_idx = 0
    for goal_cond in goal_conditions:
        action = goal_cond.copy()  # Start with goal condition (has type, obj1, obj2)
        goal_type = action['type']
        if goal_type == 'On':
            action['variant'] = _variant_for_on_action_at_step(execution_steps, step_idx)
        steps = _steps_consumed_by_goal_type(goal_type)
        step_idx += steps
        actions.append(action)
    
    # Create chained_task dict format
    chained_task = {'actions': actions}
    
    # Separate single operators from chains
    single_operators = []
    operator_chains = []
    for item in view_operators:
        if isinstance(item, list):
            # This is a chain
            operator_chains.append(item)
        else:
            # This is a single operator
            single_operators.append(item)
    
    # Check if task matches any single operator
    if single_operators and task_contains_operator(chained_task, single_operators, initial_states):
        return True
    
    # Check if task matches any operator chain
    if operator_chains and task_contains_operator_chain(chained_task, operator_chains, initial_states):
        return True
    
    return False


def delete_existing_views_for_split(
    split_name: str,
    bddl_files_base: str,
    init_states_base: str,
    suffix: Optional[str] = None,
    gen_init_states: bool = True
) -> None:
    """
    Delete all existing view directories for a split (both BDDL and init states).
    
    Args:
        split_name: Name of the split
        bddl_files_base: Base path for BDDL files (may have suffix)
        init_states_base: Base path for init states (may have suffix)
        suffix: Optional suffix for output paths
        gen_init_states: Whether to delete init state directories (if False, only delete BDDL)
    """
    # Pattern for view directories: {split_name}_*_view or {split_name}_*_inverse_view
    view_pattern = f"{split_name}_"
    
    # Delete BDDL view directories (both regular and inverse views)
    if os.path.exists(bddl_files_base):
        for item in os.listdir(bddl_files_base):
            if item.startswith(view_pattern) and (item.endswith("_view") or item.endswith("_inverse_view")):
                view_bddl_dir = os.path.join(bddl_files_base, item)
                if os.path.isdir(view_bddl_dir):
                    shutil.rmtree(view_bddl_dir)
                    print(f"Deleted existing view BDDL directory: {item}")
    
    # Delete init states view directories only if gen_init_states is True (both regular and inverse views)
    if gen_init_states and os.path.exists(init_states_base):
        for item in os.listdir(init_states_base):
            if item.startswith(view_pattern) and (item.endswith("_view") or item.endswith("_inverse_view")):
                view_init_states_dir = os.path.join(init_states_base, item)
                if os.path.isdir(view_init_states_dir):
                    shutil.rmtree(view_init_states_dir)
                    print(f"Deleted existing view init states directory: {item}")


def process_views_for_split_bddl(
    split_name: str,
    views_config: Dict,
    task_metadata_by_dir: Dict[str, Dict],
    bddl_dirs: Dict[str, str],
    bddl_files_base: str,
    init_states_base: str,
    suffix: Optional[str] = None,
    gen_init_states: bool = True
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Process views for a split: copy matching BDDL files and create filtered metadata.
    
    Args:
        split_name: Name of the split being processed
        views_config: Dict mapping view_name -> dict with:
            - 'operators': List of operator specs/chains
            - 'allow_chained_tasks': Whether to include chained tasks (default True)
            - 'also_generate_inverse_view': If True, generate inverse view with all tasks NOT in this view (default False)
            - 'exclude_second_step_tasks': If True, exclude tasks that are second-step-only derived (default True)
        task_metadata_by_dir: Dict mapping directory_name -> dict of task_name -> task_metadata
        bddl_dirs: Dict mapping directory_name -> full path to BDDL directory
        bddl_files_base: Base path for BDDL files (may have suffix)
        init_states_base: Base path for init states (may have suffix)
        suffix: Optional suffix for output paths
        gen_init_states: Whether to delete init state directories (if False, only delete BDDL)
    
    Returns:
        Dict mapping view_name -> list of (task_name, source_dir_name) tuples
        If also_generate_inverse_view is True, also includes inverse view with name "{view_name}_inverse"
    """
    # Delete all existing views for this split first
    delete_existing_views_for_split(split_name, bddl_files_base, init_states_base, suffix, gen_init_states)
    
    view_matching_tasks = {}  # Dict mapping view_name -> list of (task_name, source_dir_name) tuples
    
    for view_name, view_config in views_config.items():
        # Extract operators and allow_chained_tasks from view config
        view_operators = view_config.get('operators', [])
        allow_chained_tasks = view_config.get('allow_chained_tasks', True)
        also_generate_inverse_view = view_config.get('also_generate_inverse_view', False)
        view_filters = view_config.get('filters', {})
        duplicate_mode = view_filters.get('duplicate_mode', None)
        init_states_filters = view_filters.get('init_states', None)
        source_dirs = view_config.get('source_dirs', None)
        only_chained_tasks = view_config.get('only_chained_tasks', False)
        view_max_chain_length = view_config.get('max_chain_length', None)
        include_extra_chained_only = view_config.get('include_extra_chained_only', False)
        only_second_step_from_chain = view_config.get('only_second_step_from_chain', False)
        exclude_second_step_tasks = view_config.get('exclude_second_step_tasks', False)
        if duplicate_mode is not None and duplicate_mode != 'only_relevant_objects':
            raise ValueError(
                f"Unexpected duplicate_mode '{duplicate_mode}' in view '{view_name}'. "
                "Supported values: only_relevant_objects."
            )
        init_state_pairs = set()
        if init_states_filters is not None:
            if not isinstance(init_states_filters, list):
                raise ValueError(f"filters.init_states for view '{view_name}' must be a list.")
            for entry in init_states_filters:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(
                        f"filters.init_states entry for view '{view_name}' must be [object, location]. "
                        f"Got: {entry}"
                    )
                obj_base = get_base_object_name(entry[0])
                loc_base = get_base_object_name(entry[1])
                init_state_pairs.add((obj_base, loc_base))
        
        require_init_states_config = view_config.get('require_init_states', None)
        required_init_state_pairs = set()
        if require_init_states_config is not None:
            if not isinstance(require_init_states_config, list):
                raise ValueError(f"require_init_states for view '{view_name}' must be a list.")
            for entry in require_init_states_config:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(
                        f"require_init_states entry for view '{view_name}' must be [object, location]. "
                        f"Got: {entry}"
                    )
                obj_base = get_base_object_name(entry[0])
                loc_base = get_base_object_name(entry[1])
                required_init_state_pairs.add((obj_base, loc_base))

        match_init_states_from = view_config.get('match_init_states_from', None)
        match_init_states_targets = []  # list of (on_in_dict, open_turnon_frozenset)
        if match_init_states_from is not None:
            if not isinstance(match_init_states_from, list):
                raise ValueError(f"match_init_states_from for view '{view_name}' must be a list of task names.")
            for ref_task_name in match_init_states_from:
                found_ref = False
                for ref_dir_name, ref_bddl_dir in bddl_dirs.items():
                    ref_bddl_path = os.path.join(ref_bddl_dir, ref_task_name + '.bddl')
                    if os.path.exists(ref_bddl_path):
                        try:
                            with open(ref_bddl_path, 'r', encoding='utf-8') as f:
                                ref_content = f.read()
                            ref_parser = BDDLParser(ref_content)
                            ref_init_states = ref_parser.initial_states or {}
                            ref_open_turnon = _parse_open_turnon_from_bddl(ref_content)
                            match_init_states_targets.append((ref_init_states, ref_open_turnon))
                            found_ref = True
                        except Exception as e:
                            print(f"WARNING: Error parsing reference task {ref_task_name} for view {view_name}: {e}")
                        break
                if not found_ref:
                    print(f"WARNING: Reference task '{ref_task_name}' for match_init_states_from in view '{view_name}' not found in any BDDL directory")
        
        # Create view BDDL directory with _view suffix
        view_dir_name = f"{split_name}_{view_name}_view"
        view_bddl_dir = os.path.join(bddl_files_base, view_dir_name)
        Path(view_bddl_dir).mkdir(parents=True, exist_ok=True)
        
        matching_tasks = []  # List of (task_name, source_dir_name) tuples
        matching_task_set = set()  # Set of (task_name, source_dir_name) tuples for fast lookup
        view_task_metadata = {}  # Filtered metadata for this view
        candidate_tasks = []  # List of candidate task dicts for filtering
        
        # Track all tasks processed for inverse view generation
        all_processed_tasks = []  # List of (task_name, source_dir_name) tuples
        all_processed_task_set = set()  # Set of (task_name, source_dir_name) tuples
        
        # Iterate through all input directories
        for source_dir_name, task_metadata in task_metadata_by_dir.items():
            if source_dir_name not in bddl_dirs:
                continue
            if source_dirs is not None and source_dir_name not in source_dirs:
                continue
            
            source_bddl_dir = bddl_dirs[source_dir_name]
            
            # Iterate through all tasks in this directory
            for task_name, task_meta in task_metadata.items():
                # Load BDDL file
                bddl_file_path = os.path.join(source_bddl_dir, task_name + '.bddl')
                if not os.path.exists(bddl_file_path):
                    continue
                
                try:
                    with open(bddl_file_path, 'r', encoding='utf-8') as f:
                        bddl_content = f.read()
                    
                    # Parse goal conditions using BDDLParser
                    parser = BDDLParser(bddl_content)
                    goal_conditions = parser.goal_conditions
                    initial_states = parser.initial_states
                    
                    # Get execution_steps from metadata
                    execution_steps = task_meta.get('execution_steps', [])
                    
                    # Check if task is chained (has multiple goal conditions or multiple actions).
                    # Count action-initiating steps: Open, Turnon, Grasp, Touch each start a new
                    # logical action. Place is part of a Grasp/Touch action and is not counted.
                    # This handles tasks like open_the_top_drawer_and_put_the_bowl_inside which
                    # have 1 BDDL goal condition but 2 logical actions (open + pick-place).
                    action_initiating_prefixes = ('Open', 'Turnon', 'Grasp', 'Touch')
                    action_step_count = sum(
                        1 for step in execution_steps
                        if any(step.startswith(prefix) for prefix in action_initiating_prefixes)
                    )
                    chain_len = max(len(goal_conditions), action_step_count)
                    is_chained = chain_len > 1
                    
                    # Skip/keep tasks based on chaining configuration
                    if not allow_chained_tasks and is_chained:
                        continue
                    if only_chained_tasks and not is_chained:
                        continue
                    if view_max_chain_length is not None and chain_len > view_max_chain_length:
                        continue
                    # When include_extra_chained_only is True, from 'extra' dir only include chained tasks
                    # (exclude single-step tasks from new_environments so view/inverse only have existing + chained)
                    if include_extra_chained_only and source_dir_name == 'extra' and not is_chained:
                        continue
                    
                    # Track this task as processed (for inverse view)
                    task_key = (task_name, source_dir_name)
                    all_processed_tasks.append(task_key)
                    all_processed_task_set.add(task_key)
                    
                    # Check if task matches view operators
                    if task_matches_view_operators(goal_conditions, execution_steps, view_operators, initial_states):
                        # When requested, restrict to tasks derived as second-step-only from a chain
                        if only_second_step_from_chain:
                            derived_from_chain = task_meta.get('derived_from_chain', False)
                            derived_chain_step = task_meta.get('derived_chain_step', None)
                            if not (derived_from_chain and derived_chain_step == 2):
                                continue
                        # When requested, exclude tasks that are second-step-only derived (so only existing_env_secondstep includes them)
                        if exclude_second_step_tasks:
                            if task_meta.get('derived_from_chain', False) and task_meta.get('derived_chain_step') == 2:
                                continue

                        # Add to candidate list for filtering
                        task_meta_with_source = task_meta.copy()
                        # Determine the source split name based on source_dir_name
                        if source_dir_name == 'existing':
                            source_split = split_name
                        elif source_dir_name == 'extra':
                            source_split = f"{split_name}_extra"
                        else:
                            # Custom category (if any)
                            source_split = source_dir_name
                        task_meta_with_source['source_split'] = source_split

                        # If this is a second-step-only chain-derived task, attempt to inherit
                        # grasps_from metadata from its originating full chained task.
                        if only_second_step_from_chain and task_meta.get('derived_from_chain', False) and task_meta.get('derived_chain_step', None) == 2:
                            full_chain_task_name = task_meta.get('derived_from_full_chain_task')
                            if full_chain_task_name:
                                full_chain_meta = task_metadata.get(full_chain_task_name)
                                if full_chain_meta is not None:
                                    full_chain_grasps_from = full_chain_meta.get('grasps_from', {})
                                    if not full_chain_grasps_from:
                                        # Full chain's grasp may be covered by actions_from;
                                        # use the actions_from task itself as the grasp source.
                                        actions_from_task_name = full_chain_meta.get('actions_from')
                                        if actions_from_task_name:
                                            actions_from_meta = task_metadata.get(actions_from_task_name)
                                            if actions_from_meta is not None:
                                                inferred = {}
                                                for step in actions_from_meta.get('execution_steps', []):
                                                    stage, obj = parse_execution_step(step)
                                                    if stage == 'Grasp' and obj:
                                                        inferred[obj] = {'task': actions_from_task_name, 'object': obj}
                                                full_chain_grasps_from = inferred
                                    if full_chain_grasps_from:
                                        # Collect objects actually grasped in this second-step task
                                        grasped_objects = set()
                                        for step in execution_steps:
                                            stage, obj = parse_execution_step(step)
                                            if stage == 'Grasp' and obj:
                                                grasped_objects.add(obj)
                                        if grasped_objects:
                                            inherited_grasps = {
                                                obj: info
                                                for obj, info in full_chain_grasps_from.items()
                                                if obj in grasped_objects
                                            }
                                            if inherited_grasps:
                                                existing_grasps = task_meta_with_source.get('grasps_from', {})
                                                merged_grasps = dict(existing_grasps)
                                                # Do not overwrite any existing entries for the same object
                                                for obj, info in inherited_grasps.items():
                                                    if obj not in merged_grasps:
                                                        merged_grasps[obj] = info
                                                if merged_grasps:
                                                    task_meta_with_source['grasps_from'] = merged_grasps

                        candidate_tasks.append({
                            'task_key': task_key,
                            'task_name': task_name,
                            'source_dir_name': source_dir_name,
                            'task_meta': task_meta_with_source,
                            'bddl_file_path': bddl_file_path,
                            'bddl_content': bddl_content,
                            'execution_steps': execution_steps,
                            'initial_states': initial_states
                        })
                
                except Exception as e:
                    print(f"WARNING: Error processing task {task_name} for view {view_name}: {e}")
                    continue
        
        # Apply view filters (init_states, duplicate_mode)
        filtered_out_tasks = []
        filtered_out_task_set = set()
        filtered_candidates = []
        if init_state_pairs:
            for candidate in candidate_tasks:
                initial_states = candidate['initial_states']
                exclude = False
                for obj_name, location in initial_states.items():
                    obj_base = get_base_object_name(obj_name)
                    loc_base = get_base_object_name(location)
                    if (obj_base, loc_base) in init_state_pairs:
                        exclude = True
                        break
                if exclude:
                    filtered_out_tasks.append(candidate['task_key'])
                    filtered_out_task_set.add(candidate['task_key'])
                else:
                    filtered_candidates.append(candidate)
        else:
            filtered_candidates = candidate_tasks
        
        if required_init_state_pairs:
            require_passed = []
            for candidate in filtered_candidates:
                initial_states = candidate['initial_states']
                actual_pairs = set()
                for obj_name, location in initial_states.items():
                    actual_pairs.add((get_base_object_name(obj_name), get_base_object_name(location)))
                if required_init_state_pairs.issubset(actual_pairs):
                    require_passed.append(candidate)
                else:
                    filtered_out_tasks.append(candidate['task_key'])
                    filtered_out_task_set.add(candidate['task_key'])
            filtered_candidates = require_passed

        if match_init_states_targets:
            exact_match_passed = []
            for candidate in filtered_candidates:
                init_states_candidate = candidate['initial_states'] or {}
                open_turnon_candidate = _parse_open_turnon_from_bddl(candidate['bddl_content'])
                keep = False
                for ref_init_states, ref_open_turnon in match_init_states_targets:
                    if init_states_candidate == ref_init_states and open_turnon_candidate == ref_open_turnon:
                        keep = True
                        break
                if keep:
                    exact_match_passed.append(candidate)
                else:
                    filtered_out_tasks.append(candidate['task_key'])
                    filtered_out_task_set.add(candidate['task_key'])
            filtered_candidates = exact_match_passed

        final_candidates = []
        if duplicate_mode == 'only_relevant_objects':
            for candidate in filtered_candidates:
                is_duplicate = False
                for kept in final_candidates:
                    if only_relevant_entities_and_locations_differ(
                        kept['bddl_content'],
                        candidate['bddl_content'],
                        candidate['execution_steps']
                    ):
                        is_duplicate = True
                        break
                if is_duplicate:
                    filtered_out_tasks.append(candidate['task_key'])
                    filtered_out_task_set.add(candidate['task_key'])
                else:
                    final_candidates.append(candidate)
        else:
            final_candidates = filtered_candidates

        filtered_labels = [f"{name} ({source})" for name, source in filtered_out_tasks]
        print(f"View '{view_name}': filtered out {len(filtered_labels)} tasks")
        for label in filtered_labels:
            print(f"  - {label}")

        # Copy BDDL files and build metadata for kept tasks
        for candidate in final_candidates:
            task_key = candidate['task_key']
            task_name = candidate['task_name']
            # Copy BDDL file to view directory
            dest_bddl_path = os.path.join(view_bddl_dir, task_name + '.bddl')
            shutil.copy2(candidate['bddl_file_path'], dest_bddl_path)
            # Add to matching tasks
            matching_tasks.append(task_key)
            matching_task_set.add(task_key)
            # Add to filtered metadata with source split information
            view_task_metadata[task_name] = candidate['task_meta']

        # Save filtered task_metadata.yaml
        if view_task_metadata:
            metadata_to_save = {
                'base_split_name': split_name,
                'tasks': view_task_metadata
            }
            metadata_path = os.path.join(view_bddl_dir, 'task_metadata.yaml')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
        else:
            # Create empty metadata file if no tasks matched
            metadata_to_save = {
                'base_split_name': split_name,
                'tasks': {}
            }
            metadata_path = os.path.join(view_bddl_dir, 'task_metadata.yaml')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
        
        view_matching_tasks[view_name] = matching_tasks
        print(f"View '{view_name}': {len(matching_tasks)} matching tasks")
        
        # Generate inverse view if requested
        if also_generate_inverse_view:
            inverse_view_name = f"{view_name}_inverse"
            inverse_view_dir_name = f"{split_name}_{inverse_view_name}_view"
            inverse_view_bddl_dir = os.path.join(bddl_files_base, inverse_view_dir_name)
            Path(inverse_view_bddl_dir).mkdir(parents=True, exist_ok=True)
            
            inverse_matching_tasks = []  # List of (task_name, source_dir_name) tuples
            inverse_view_task_metadata = {}  # Filtered metadata for inverse view
            
            # Find all tasks that were NOT in the original view
            for task_key in all_processed_tasks:
                if task_key not in matching_task_set and task_key not in filtered_out_task_set:
                    task_name, source_dir_name = task_key
                    inverse_matching_tasks.append(task_key)
                    
                    # Copy BDDL file to inverse view directory
                    source_bddl_dir = bddl_dirs[source_dir_name]
                    bddl_file_path = os.path.join(source_bddl_dir, task_name + '.bddl')
                    dest_bddl_path = os.path.join(inverse_view_bddl_dir, task_name + '.bddl')
                    shutil.copy2(bddl_file_path, dest_bddl_path)
                    
                    # Add to filtered metadata with source split information
                    task_meta = task_metadata_by_dir[source_dir_name][task_name]
                    task_meta_with_source = task_meta.copy()
                    # Determine the source split name based on source_dir_name
                    if source_dir_name == 'existing':
                        source_split = split_name
                    elif source_dir_name == 'extra':
                        source_split = f"{split_name}_extra"
                    else:
                        # Custom category (if any)
                        source_split = source_dir_name
                    task_meta_with_source['source_split'] = source_split
                    inverse_view_task_metadata[task_name] = task_meta_with_source
            
            # Save filtered task_metadata.yaml for inverse view
            if inverse_view_task_metadata:
                metadata_to_save = {
                    'base_split_name': split_name,
                    'tasks': inverse_view_task_metadata
                }
                metadata_path = os.path.join(inverse_view_bddl_dir, 'task_metadata.yaml')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
            else:
                # Create empty metadata file if no tasks matched
                metadata_to_save = {
                    'base_split_name': split_name,
                    'tasks': {}
                }
                metadata_path = os.path.join(inverse_view_bddl_dir, 'task_metadata.yaml')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
            
            view_matching_tasks[inverse_view_name] = inverse_matching_tasks
            print(f"Inverse view '{inverse_view_name}': {len(inverse_matching_tasks)} matching tasks")
    
    return view_matching_tasks


def load_metadata_from_files(
    split_name: str,
    bddl_files_base: str,
    suffix: Optional[str] = None,
    views_config: Optional[Dict] = None
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Load task metadata from existing task_metadata.yaml files.
    
    Args:
        split_name: Name of the split
        bddl_files_base: Base path for BDDL files (may have suffix)
        suffix: Optional suffix for output paths
        views_config: Optional views config to exclude view directories
    
    Returns:
        Tuple of (task_metadata_by_dir, bddl_dirs)
        - task_metadata_by_dir: Dict mapping directory_name -> dict of task_name -> task_metadata
        - bddl_dirs: Dict mapping directory_name -> full path to BDDL directory
    """
    task_metadata_by_dir = {}
    bddl_dirs = {}
    
    # Get view directory names to exclude (with _view suffix)
    view_dir_names = set()
    if views_config:
        for view_name in views_config.keys():
            view_dir_names.add(f"{split_name}_{view_name}_view")
    
    # Load existing tasks metadata
    existing_bddl_dir = os.path.join(bddl_files_base, split_name)
    existing_metadata_path = os.path.join(existing_bddl_dir, 'task_metadata.yaml')
    if os.path.exists(existing_metadata_path):
        with open(existing_metadata_path, 'r', encoding='utf-8') as f:
            existing_metadata = yaml.safe_load(f) or {}
        task_metadata_by_dir['existing'] = existing_metadata.get('tasks', {})
        bddl_dirs['existing'] = existing_bddl_dir
    
    # Load extra tasks metadata
    extra_bddl_dir = os.path.join(bddl_files_base, f'{split_name}_extra')
    extra_metadata_path = os.path.join(extra_bddl_dir, 'task_metadata.yaml')
    if os.path.exists(extra_metadata_path):
        with open(extra_metadata_path, 'r', encoding='utf-8') as f:
            extra_metadata = yaml.safe_load(f) or {}
        task_metadata_by_dir['extra'] = extra_metadata.get('tasks', {})
        bddl_dirs['extra'] = extra_bddl_dir
    
    return task_metadata_by_dir, bddl_dirs


def copy_init_states_for_views(
    split_name: str,
    views_config: Dict,
    view_matching_tasks: Dict[str, List[Tuple[str, str]]],
    init_states_dirs: Dict[str, str],
    init_states_base: str,
    suffix: Optional[str] = None,
    gen_init_states: bool = True
) -> None:
    """
    Copy init state files for tasks that match views.
    
    Args:
        split_name: Name of the split
        views_config: Views configuration
        view_matching_tasks: Dict mapping view_name -> list of (task_name, source_dir_name) tuples
        init_states_dirs: Dict mapping directory_name -> full path to init states directory
        init_states_base: Base path for init states (may have suffix)
        suffix: Optional suffix for output paths
        gen_init_states: Whether to copy init states (if False, function returns early)
    """
    # Skip copying init states if gen_init_states is False
    if not gen_init_states:
        return
    # For existing tasks, use default (non-suffixed) init states path if suffix is provided
    # because existing tasks' init states are not copied to suffixed directories
    default_init_states_base = get_libero_path('init_states')
    
    for view_name, matching_tasks in view_matching_tasks.items():
        if not matching_tasks:
            continue
        
        # Create view init states directory with _view suffix
        view_dir_name = f"{split_name}_{view_name}_view"
        view_init_states_dir = os.path.join(init_states_base, view_dir_name)
        Path(view_init_states_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy init state files for each matching task
        for task_name, source_dir_name in matching_tasks:
            # For existing tasks, use default (non-suffixed) path
            if source_dir_name == 'existing' and suffix is not None:
                source_init_states_dir = os.path.join(default_init_states_base, split_name)
            elif source_dir_name not in init_states_dirs:
                print(f"WARNING: Source directory '{source_dir_name}' not found in init_states_dirs for task {task_name}")
                continue
            else:
                source_init_states_dir = init_states_dirs[source_dir_name]
            
            # Init state file pattern: {task_name}.pruned_init
            init_state_filename = f"{task_name}.pruned_init"
            source_init_state_path = os.path.join(source_init_states_dir, init_state_filename)
            
            if not os.path.exists(source_init_state_path):
                print(f"WARNING: Init state file not found: {source_init_state_path}")
                continue
            
            # Copy init state file to view directory
            dest_init_state_path = os.path.join(view_init_states_dir, init_state_filename)
            shutil.copy2(source_init_state_path, dest_init_state_path)
        
        print(f"Copied init states for view '{view_name}': {len(matching_tasks)} files")


def should_skip_task_generation(
    chained_task: Dict,
    skip_conditions: List[Dict],
    cur_metadata: Dict,
    execution_steps: List[str],
    existing_tasks_metadata: Optional[Dict] = None,
    initial_states: Optional[Dict[str, str]] = None,
    open_predicates: Optional[frozenset] = None,
) -> bool:
    """
    Check if a task should be skipped based on skip_generation_for conditions.

    Args:
        chained_task: The task specification dict with 'actions' key containing list of actions
        skip_conditions: List of skip condition dictionaries from skip_generation_for config
        cur_metadata: The current task's metadata dictionary
        execution_steps: List of execution step strings for the current task
        existing_tasks_metadata: Optional dict of existing task metadata (needed to compute actions_from_steps)
        open_predicates: Optional frozenset of ("Open"|"Turnon", obj_name) tuples from the init section

    Returns:
        True if the task should be skipped (matches any condition), False otherwise
    """
    if not skip_conditions:
        return False
    
    # Get actions_from value from metadata
    actions_from = cur_metadata.get('actions_from')
    
    # Compute actions_from_steps if actions_from is present but actions_from_steps is not yet in metadata
    actions_from_steps = cur_metadata.get('actions_from_steps')
    if actions_from is not None and actions_from_steps is None and existing_tasks_metadata is not None:
        if actions_from in existing_tasks_metadata:
            from_task_metadata = existing_tasks_metadata[actions_from]
            steps_in_from_task = from_task_metadata.get('execution_steps', [])
            
            # Compare steps sequentially and include all matching steps (same logic as later in the code)
            matching_steps = []
            min_len = min(len(execution_steps), len(steps_in_from_task))
            
            for i in range(min_len):
                cur_step = execution_steps[i]
                from_step = steps_in_from_task[i]
                
                cur_stage, cur_obj = parse_execution_step(cur_step)
                from_stage, from_obj = parse_execution_step(from_step)
                
                # Steps must have the same stage
                if cur_stage != from_stage:
                    break
                
                # For Place steps, the format is "Place obj1 obj2" where obj2 is the target
                if cur_stage == 'Place':
                    cur_parts = cur_step.split()
                    from_parts = from_step.split()
                    if len(cur_parts) >= 3 and len(from_parts) >= 3:
                        cur_obj1 = cur_parts[1]  # object being placed
                        cur_obj2 = cur_parts[2]  # target location
                        from_obj1 = from_parts[1]
                        from_obj2 = from_parts[2]
                        
                        # Compare by exact instance: both object and target must match exactly
                        if cur_obj1 == from_obj1 and cur_obj2 == from_obj2:
                            matching_steps.append(from_step)
                        else:
                            # Place step doesn't match, stop here
                            break
                    else:
                        # Malformed Place step, stop here
                        break
                elif cur_obj and from_obj:
                    # For other steps (Grasp, Open, etc.), compare by exact instance
                    if cur_obj == from_obj:
                        matching_steps.append(from_step)
                    else:
                        # Steps don't match, stop here
                        break
                else:
                    # Steps without objects (shouldn't happen in practice, but handle gracefully)
                    matching_steps.append(from_step)
            
            actions_from_steps = matching_steps
    
    for condition in skip_conditions:
        condition_matched = True
        
        # Check in_execution_steps condition
        if 'in_execution_steps' in condition:
            required_action_types = condition['in_execution_steps']
            if not isinstance(required_action_types, list):
                required_action_types = [required_action_types]
            
            # Check if any of the required action types appear in the execution steps
            execution_steps_contain_required_type = False
            for step in execution_steps:
                step_stage, _ = parse_execution_step(step)
                if step_stage in required_action_types:
                    execution_steps_contain_required_type = True
                    break
            
            if not execution_steps_contain_required_type:
                condition_matched = False
        
        # Check not_in_actions_from_steps condition
        if 'not_in_actions_from_steps' in condition:
            required_action_types = condition['not_in_actions_from_steps']
            if not isinstance(required_action_types, list):
                required_action_types = [required_action_types]
            
            # Condition passes (skip task) if:
            # 1. actions_from_steps is not present in metadata, OR
            # 2. actions_from_steps is present but does not contain any of the listed action types
            not_in_actions_from_steps_passes = False
            if actions_from_steps is None or len(actions_from_steps) == 0:
                # actions_from_steps is not present, condition passes
                not_in_actions_from_steps_passes = True
            else:
                # Check if any of the required action types are in actions_from_steps
                # Extract action types from the step strings (e.g., "Open drawer_1" -> "Open")
                steps_contain_required_type = False
                for step in actions_from_steps:
                    step_stage, _ = parse_execution_step(step)
                    if step_stage in required_action_types:
                        steps_contain_required_type = True
                        break
                
                # If steps do NOT contain any required type, condition passes (skip)
                if not steps_contain_required_type:
                    not_in_actions_from_steps_passes = True
            
            # If this condition doesn't pass, mark the overall condition as not matched
            if not not_in_actions_from_steps_passes:
                condition_matched = False
        
        # Check init_state_contains condition
        if 'init_state_contains' in condition:
            spec = condition['init_state_contains']
            obj_name = spec.get('object')
            loc_name = spec.get('location')
            if initial_states is None or obj_name is None or loc_name is None:
                condition_matched = False
            else:
                actual_loc = initial_states.get(obj_name)
                if actual_loc != loc_name:
                    condition_matched = False

        # Check action_at_position condition
        # Checks if the action at a given position in the chain matches an operator spec.
        # Uses the same operator/source/target matching as action_matches_operator_spec.
        # Supports optional min_chain_length to only match chains of a minimum length.
        # Example: {position: -1, min_chain_length: 2, operator: can_place_on, source: wine_bottle, target: wine_rack_top_region}
        if 'action_at_position' in condition:
            spec = condition['action_at_position']
            position = spec.get('position', -1)
            min_chain_length = spec.get('min_chain_length')
            actions = chained_task.get('actions', [])
            if not actions:
                condition_matched = False
            elif min_chain_length is not None and len(actions) < min_chain_length:
                condition_matched = False
            else:
                try:
                    action = actions[position]
                except IndexError:
                    condition_matched = False
                else:
                    operator_spec = {k: v for k, v in spec.items() if k not in ('position', 'min_chain_length')}
                    if not action_matches_operator_spec(action, operator_spec):
                        condition_matched = False

        # Check init_open_contains condition
        # Skips tasks whose init section contains an Open predicate for the specified object.
        # Use the BDDL instance name (e.g., wooden_cabinet_1_middle_region).
        if 'init_open_contains' in condition:
            obj_name = condition['init_open_contains']
            if open_predicates is None or ("Open", obj_name) not in open_predicates:
                condition_matched = False

        # If all conditions in this entry are matched, skip the task
        if condition_matched:
            return True

    return False

def task_contains_operator(chained_task: Dict, operators: List[Dict], initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if any action in a chained task matches any operator specification.
    
    Args:
        chained_task: Dict with 'actions' list
        operators: List of operator specs
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if any action matches any operator spec
    """
    for action in chained_task['actions']:
        for operator_spec in operators:
            if action_matches_operator_spec(action, operator_spec, initial_states):
                return True
    return False




def task_matches_operator_chain(chained_task: Dict, operator_chain: List[Dict], initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a chained task matches an operator chain (operators in a specific order).
    
    This checks if the task's actions contain a subsequence that matches the chain in order.
    For example, if chain is [turn_on_stove, place_bowl_on_stove], this checks if the task
    has actions that match these operators in that order (not necessarily consecutive).
    
    Args:
        chained_task: Dict with 'actions' list
        operator_chain: List of operator specs in order
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if the task's actions match the chain in order
    """
    actions = chained_task['actions']
    chain_idx = 0  # Index into the operator chain
    
    # Try to match each operator in the chain sequentially
    for action in actions:
        if chain_idx >= len(operator_chain):
            # Matched all operators in the chain
            return True
        
        # Check if current action matches the current operator in the chain
        current_operator = operator_chain[chain_idx]
        if action_matches_operator_spec(action, current_operator, initial_states):
            # Matched this operator, move to next in chain
            chain_idx += 1
    
    # Check if we matched all operators in the chain
    return chain_idx >= len(operator_chain)


def task_matches_pattern(chained_task: Dict, pattern: Dict, initial_states: Dict[str, str]) -> bool:
    """
    Check if a chained task matches a pattern specification.
    
    Args:
        chained_task: Dict with 'actions' list
        pattern: Dict with keys:
            - grasped_object: base name of object being grasped (optional)
            - initial_location: base name of initial location, or list of base names (optional)
                If a list, matches if any value in the list matches
            - placed_object: base name of object being placed (optional)
            - place_location: base name of place location, or list of base names (optional)
                If a list, matches if any value in the list matches.
                Matches either:
                1. Direct placement on the location (e.g., placing on main_table_next_to_box_region)
                2. Placement on an object that is initially located at the location
                   (e.g., placing on a bowl that is on main_table_next_to_box_region)
        initial_states: Dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if the task matches the pattern
    """
    actions = chained_task['actions']
    if not actions:
        return False
    
    # For single-action On tasks, check the pattern
    if len(actions) == 1 and actions[0]['type'] == 'On' and actions[0].get('variant', 'grasp') == 'grasp':
        action = actions[0]
        obj1_base = get_base_object_name(action['obj1'])
        obj2_base = get_base_object_name(action['obj2'])
        
        # Check grasped_object
        if 'grasped_object' in pattern:
            if obj1_base != get_base_object_name(pattern['grasped_object']):
                return False
        
        # Check initial_location (strip _resized suffix for comparison since resized regions are functionally equivalent)
        # Supports both single value and list of values (matches if any value in list matches)
        if 'initial_location' in pattern:
            obj1_initial_location = initial_states.get(action['obj1'])
            if obj1_initial_location is None:
                return False
            obj1_initial_location_base = get_base_object_name(obj1_initial_location)
            obj1_initial_location_base_stripped = _strip_resized_suffix(obj1_initial_location_base)
            
            pattern_initial_location = pattern['initial_location']
            # Support both single value and list of values
            if isinstance(pattern_initial_location, list):
                # Check if any value in the list matches
                matches = False
                for loc in pattern_initial_location:
                    pattern_initial_location_base = get_base_object_name(loc)
                    if obj1_initial_location_base_stripped == pattern_initial_location_base:
                        matches = True
                        break
                if not matches:
                    return False
            else:
                # Single value (original behavior)
                pattern_initial_location_base = get_base_object_name(pattern_initial_location)
                if obj1_initial_location_base_stripped != pattern_initial_location_base:
                    return False
        
        # Check placed_object (should be same as grasped_object for On tasks)
        if 'placed_object' in pattern:
            if obj1_base != get_base_object_name(pattern['placed_object']):
                return False
        
        # Check place_location (strip _resized suffix for comparison since resized regions are functionally equivalent)
        # Supports both single value and list of values (matches if any value in list matches)
        # Also supports matching when placing on an object that is initially located at the specified location
        if 'place_location' in pattern:
            obj2_base_stripped = _strip_resized_suffix(obj2_base)
            
            pattern_place_location = pattern['place_location']
            # Support both single value and list of values
            pattern_locations = pattern_place_location if isinstance(pattern_place_location, list) else [pattern_place_location]
            
            matches = False
            for loc in pattern_locations:
                pattern_place_location_base = get_base_object_name(loc)
                
                # Direct match: place_location directly matches obj2 (e.g., placing on main_table_next_to_box_region)
                if obj2_base_stripped == pattern_place_location_base:
                    matches = True
                    break
                
                # Indirect match: obj2 is an object that is initially located at the pattern location
                # (e.g., placing on a bowl that is on main_table_next_to_box_region)
                obj2_initial_location = initial_states.get(action['obj2'])
                if obj2_initial_location is not None:
                    obj2_initial_location_base = get_base_object_name(obj2_initial_location)
                    obj2_initial_location_base_stripped = _strip_resized_suffix(obj2_initial_location_base)
                    if obj2_initial_location_base_stripped == pattern_place_location_base:
                        matches = True
                        break
            
            if not matches:
                return False
        
        return True

    # For multi-action tasks, check if any On/grasp action in the chain matches the pattern.
    # This allows patterns like {grasped_object, place_location} to match chained tasks
    # (e.g., Open + place wine bottle on cabinet top).
    for action in actions:
        if action['type'] != 'On' or action.get('variant', 'grasp') != 'grasp':
            continue

        obj1_base = get_base_object_name(action['obj1'])
        obj2_base = get_base_object_name(action['obj2'])

        if 'grasped_object' in pattern:
            if obj1_base != get_base_object_name(pattern['grasped_object']):
                continue

        if 'placed_object' in pattern:
            if obj1_base != get_base_object_name(pattern['placed_object']):
                continue

        if 'initial_location' in pattern:
            obj1_initial_location = initial_states.get(action['obj1'])
            if obj1_initial_location is None:
                continue
            obj1_initial_location_base_stripped = _strip_resized_suffix(get_base_object_name(obj1_initial_location))
            pattern_initial_location = pattern['initial_location']
            pattern_locations = pattern_initial_location if isinstance(pattern_initial_location, list) else [pattern_initial_location]
            if not any(_strip_resized_suffix(get_base_object_name(loc)) == obj1_initial_location_base_stripped for loc in pattern_locations):
                continue

        if 'place_location' in pattern:
            obj2_base_stripped = _strip_resized_suffix(obj2_base)
            pattern_place_location = pattern['place_location']
            pattern_locations = pattern_place_location if isinstance(pattern_place_location, list) else [pattern_place_location]
            matched = False
            for loc in pattern_locations:
                pattern_place_location_base = get_base_object_name(loc)
                if obj2_base_stripped == pattern_place_location_base:
                    matched = True
                    break
                obj2_initial_location = initial_states.get(action['obj2'])
                if obj2_initial_location is not None:
                    obj2_initial_location_base_stripped = _strip_resized_suffix(get_base_object_name(obj2_initial_location))
                    if obj2_initial_location_base_stripped == pattern_place_location_base:
                        matched = True
                        break
            if not matched:
                continue

        # All specified pattern keys matched for this action
        return True

    return False


def task_contains_operator_chain(chained_task: Dict, operator_chains: List[List[Dict]], initial_states: Optional[Dict[str, str]] = None) -> bool:
    """
    Check if a chained task matches any operator chain.
    
    Args:
        chained_task: Dict with 'actions' list
        operator_chains: List of operator chains (each chain is a list of operator specs in order)
        initial_states: Optional dict mapping object_name -> location_name from BDDL init section
    
    Returns:
        True if the task matches any operator chain
    """
    for operator_chain in operator_chains:
        if task_matches_operator_chain(chained_task, operator_chain, initial_states):
            return True
    return False


def is_chain_reasonable(
    prev_action: Dict,
    next_action: Dict,
    objects_config: Dict,
    operators_config: Dict
) -> bool:
    """
    Check if chaining next_action after prev_action is reasonable.
    
    prev_action's object MUST have reasonable_next_involves constraints defined,
    and next_action must match at least one of those {object, operator} pairs.
    
    If no reasonable_next_involves is defined, returns False (no chaining allowed).
    """
    # Get the primary object from prev_action
    prev_type = prev_action['type']
    if prev_type == 'On':
        prev_obj = prev_action['obj2']  # The target location
    elif prev_type == 'In':
        prev_obj = prev_action['obj1']  # The container
    elif prev_type == 'Open':
        prev_obj = prev_action['obj1']
    elif prev_type == 'Turnon':
        prev_obj = prev_action['obj1']
    else:
        return False
    
    # Check if this object has reasonable_next_involves constraints
    # Use get_object_config to handle base name matching
    obj_config = get_object_config(prev_obj, objects_config)
    if obj_config is None:
        return False
    
    chaining = obj_config.get('chaining', {})
    reasonable_next = chaining.get('reasonable_next_involves')
    
    if reasonable_next is None:
        # No constraint defined - chaining not allowed when reasonable_only is true
        return False
    
    # Get all (object, operator) pairs in the next action
    next_pairs = get_task_object_operator_pairs(next_action)
    
    # Check if any pair matches a constraint
    # Handle base name matching for both constraint and task objects
    for constraint in reasonable_next:
        required_obj = constraint['object']
        required_op = constraint['operator']
        required_obj_base = get_base_object_name(required_obj)
        
        for obj, op in next_pairs:
            obj_base = get_base_object_name(obj)
            # Match by exact name or base name
            obj_matches = (obj == required_obj or obj == required_obj_base or 
                          obj_base == required_obj or obj_base == required_obj_base)
            if obj_matches and op == required_op:
                return True
    
    return False


def generate_chained_task_combinations(
    objects_in_bddl: List[str],
    objects_config: Dict,
    operators_config: Dict,
    chaining_config: Dict
) -> List[Dict]:
    """
    Generate all valid chained task combinations.
    
    This function generates multi-action tasks by:
    1. Starting with single-action tasks
    2. For each valid chain length, extending chains with valid next actions
    3. Respecting ordering constraints (action replay must be first)
    4. Tracking world state to ensure logical combinations
    5. If reasonable_only is true, filtering by reasonable_next_involves constraints
    
    Args:
        objects_in_bddl: List of objects available in the BDDL
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
        chaining_config: Chaining configuration (max_chain_length, reasonable_only)
    
    Returns:
        List of chained task specifications (each with 'actions' list)
    """
    max_chain_length = chaining_config.get('max_chain_length', 2)
    reasonable_only = chaining_config.get('reasonable_only', False)
    exclude_first_action_types = set(chaining_config.get('exclude_first_action_types', []))
    
    # Get all single-action tasks
    single_tasks = generate_operator_combinations(objects_in_bddl, objects_config, operators_config)
    
    # Wrap single tasks in the chained format
    all_chained_tasks = []
    
    # Include single-action tasks
    for task in single_tasks:
        all_chained_tasks.append({
            'actions': [task],
            'is_chained': False
        })
    
    if max_chain_length < 2:
        return all_chained_tasks
    
    # Generate chained tasks
    # Start with tasks that can be first in a chain
    initial_chains = []
    for task in single_tasks:
        if task['type'] in exclude_first_action_types:
            continue
        world_state = WorldState()
        if is_action_valid(task, world_state, objects_config, operators_config, is_first_action=True):
            new_state = apply_action(task, world_state, objects_config, operators_config)
            initial_chains.append({
                'actions': [task],
                'world_state': new_state
            })
    
    # Extend chains up to max_chain_length
    current_chains = initial_chains
    for chain_len in range(2, max_chain_length + 1):
        next_chains = []
        
        for chain in current_chains:
            current_state = chain['world_state']
            current_actions = chain['actions']
            prev_action = current_actions[-1]
            
            # Try to extend with each single task
            for next_task in single_tasks:
                if not is_action_valid(next_task, current_state, objects_config, operators_config, is_first_action=False):
                    continue
                
                # Check reasonable constraint if enabled
                if reasonable_only and not is_chain_reasonable(prev_action, next_task, objects_config, operators_config):
                    continue
                
                # Create new chain
                new_actions = current_actions + [next_task]
                new_state = apply_action(next_task, current_state, objects_config, operators_config)
                
                next_chains.append({
                    'actions': new_actions,
                    'world_state': new_state
                })
        
        # Add chains of this length to results
        for chain in next_chains:
            all_chained_tasks.append({
                'actions': chain['actions'],
                'is_chained': True
            })
        
        current_chains = next_chains
    
    return all_chained_tasks


def generate_chained_filename_and_lang(
    chained_task: Dict,
    objects_config: Dict,
    operators_config: Dict,
    naming_convention: Optional[Dict] = None,
    initial_location: Optional[str] = None,
    initial_location_relation: Optional[str] = None,
    target_initial_location: Optional[str] = None,
    target_initial_location_relation: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate filename and BDDL language for a chained task.
    
    For chained tasks, joins individual action descriptions with "and_then" / "and then".
    
    Args:
        chained_task: Dictionary with 'actions' list
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
        naming_convention: Optional naming convention overrides (from split config)
        initial_location: Optional initial location of first action's obj1 (from base BDDL init)
        initial_location_relation: Optional relation type ('On' or 'In') for initial location
        target_initial_location: Optional initial location of first action's obj2 (target object) from base BDDL init
        target_initial_location_relation: Optional relation type ('On' or 'In') for target initial location
    
    Returns:
        Tuple of (filename, bddl_language)
    """
    actions = chained_task['actions']
    
    if len(actions) == 1:
        return generate_filename_and_lang(actions[0], objects_config, operators_config, naming_convention, initial_location, initial_location_relation, target_initial_location, target_initial_location_relation)
    
    # Generate filename and language for each action
    filenames = []
    bddl_langs = []
    
    # For chained tasks, only use initial_location and target_initial_location for the first action
    for i, action in enumerate(actions):
        action_initial_location = initial_location if i == 0 else None
        action_initial_location_relation = initial_location_relation if i == 0 else None
        action_target_initial_location = target_initial_location if i == 0 else None
        action_target_initial_location_relation = target_initial_location_relation if i == 0 else None
        filename, bddl_lang = generate_filename_and_lang(action, objects_config, operators_config, naming_convention, action_initial_location, action_initial_location_relation, action_target_initial_location, action_target_initial_location_relation)
        filenames.append(filename)
        bddl_langs.append(bddl_lang)
    
    # Join with "and_then" for filename, "and then" for language
    combined_filename = "_and_then_".join(filenames)
    combined_bddl_lang = " and then ".join(bddl_langs)
    
    return combined_filename, combined_bddl_lang


def generate_chained_execution_steps(
    chained_task: Dict,
    objects_config: Dict,
    operators_config: Dict
) -> List[str]:
    """
    Generate execution steps for a chained task.
    
    Concatenates execution steps from all actions in the chain.
    
    Args:
        chained_task: Dictionary with 'actions' list
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
    
    Returns:
        List of execution steps
    """
    all_steps = []
    
    for action in chained_task['actions']:
        steps = generate_execution_steps(action, objects_config, operators_config)
        all_steps.extend(steps)
    
    return all_steps


def generate_chained_bddl_goal(
    chained_task: Dict,
    objects_config: Dict,
    operators_config: Dict
) -> Tuple[str, List[str]]:
    """
    Generate BDDL goal expression and objects of interest for a chained task.
    
    IMPORTANT: This function uses the object names directly from the action specs,
    which should be specific instance names (e.g., akita_black_bowl_1, akita_black_bowl_2),
    NOT base names (e.g., akita_black_bowl). The BDDL file requires specific instances.
    
    Args:
        chained_task: Dictionary with 'actions' list (actions contain instance names like obj1='akita_black_bowl_1')
        objects_config: Object configuration from YAML (may contain base names)
        operators_config: Operator configuration from YAML
    
    Returns:
        Tuple of (bddl_goal_expression, objects_of_interest)
        Both use specific instance names from the action specs.
    """
    goal_parts = []
    objects_of_interest = []
    seen_objects = set()
    
    for action in chained_task['actions']:
        task_type = action['type']
        
        if task_type == 'On':
            # Use instance names directly from action specs (e.g., 'akita_black_bowl_1', not 'akita_black_bowl')
            obj1 = action['obj1']  # Instance name like 'akita_black_bowl_1'
            obj2 = action['obj2']  # Instance name like 'plate_1'
            goal_parts.append(f"(On {obj1} {obj2})")
            for obj in [obj1, obj2]:
                if obj not in seen_objects:
                    objects_of_interest.append(obj)  # Store instance name
                    seen_objects.add(obj)
                    
        elif task_type in ('In', 'PlaceIn'):
            # Use instance names directly from action specs
            container = action['obj1']  # Instance name
            obj_to_place = action['obj2']  # Instance name
            goal_parts.append(f"(In {obj_to_place} {container})")
            for obj in [obj_to_place, container]:
                if obj not in seen_objects:
                    objects_of_interest.append(obj)  # Store instance name
                    seen_objects.add(obj)

        elif task_type == 'Open':
            # Use instance name directly from action spec
            obj = action['obj1']  # Instance name
            goal_parts.append(f"(Open {obj})")
            if obj not in seen_objects:
                objects_of_interest.append(obj)  # Store instance name
                seen_objects.add(obj)
                
        elif task_type == 'Turnon':
            # Use instance name directly from action spec
            obj = action['obj1']  # Instance name
            goal_parts.append(f"(Turnon {obj})")
            if obj not in seen_objects:
                objects_of_interest.append(obj)  # Store instance name
                seen_objects.add(obj)
    
    # Combine goal parts with And
    if len(goal_parts) == 1:
        bddl_goal = f"(And {goal_parts[0]})"
    else:
        bddl_goal = "(And " + " ".join(goal_parts) + ")"
    
    return bddl_goal, objects_of_interest


def generate_filename_and_lang(
    task_spec: Dict,
    objects_config: Dict,
    operators_config: Dict,
    naming_convention: Optional[Dict] = None,
    initial_location: Optional[str] = None,
    initial_location_relation: Optional[str] = None,
    target_initial_location: Optional[str] = None,
    target_initial_location_relation: Optional[str] = None
) -> Tuple[str, str]:
    """Generate filename and BDDL language for a task specification
    
    Args:
        task_spec: Task specification dict
        objects_config: Object configuration from YAML
        operators_config: Operator configuration from YAML
        naming_convention: Optional naming convention overrides (from split config)
        initial_location: Optional initial location of obj1 (from base BDDL init)
        initial_location_relation: Optional relation type ('On' or 'In') for initial location
        target_initial_location: Optional initial location of obj2 (target object) from base BDDL init
        target_initial_location_relation: Optional relation type ('On' or 'In') for target initial location
    """
    task_type = task_spec['type']
    
    if task_type == 'On':
        variant = task_spec['variant']
        obj1_name = task_spec['obj1']
        obj2_name = task_spec['obj2']
        
        # Get configs, matching by base name if needed
        obj1_config = get_object_config(obj1_name, objects_config)
        obj2_config = get_object_config(obj2_name, objects_config)
        
        if obj1_config is None or obj2_config is None:
            # Provide helpful error message showing both instance names and base names we tried to match
            obj1_base = get_base_object_name(obj1_name)
            obj2_base = get_base_object_name(obj2_name)
            obj1_tried = [obj1_name] if obj1_name != obj1_base else [obj1_name]
            obj2_tried = [obj2_name] if obj2_name != obj2_base else [obj2_name]
            if obj1_name != obj1_base:
                obj1_tried.append(obj1_base)
            if obj2_name != obj2_base:
                obj2_tried.append(obj2_base)
            
            missing_objs = []
            if obj1_config is None:
                missing_objs.append(f"{obj1_name} (tried base: {obj1_base})")
            if obj2_config is None:
                missing_objs.append(f"{obj2_name} (tried base: {obj2_base})")
            
            available_keys = list(objects_config.keys())[:10]  # Show first 10 for debugging
            raise ValueError(
                f"Objects not in config: {', '.join(missing_objs)}. "
                f"Available config keys (first 10): {available_keys}"
            )
        
        if variant == 'push':
            # push_the_plate_to_the_front_of_the_stove
            obj1_filename = obj1_config['object_filename_lang']
            obj2_filename = obj2_config['object_filename_lang']
            # Location prefix is "to_the" for push operations
            filename = f"push_the_{obj1_filename}_to_the_{obj2_filename}"
            
            obj1_bddl = obj1_config['object_bddl_lang']
            obj2_bddl = obj2_config['object_bddl_lang']
            bddl_lang = f"Push the {obj1_bddl} to the {obj2_bddl}"
        else:
            # Check if we should use naming convention override
            use_naming_convention = naming_convention and naming_convention.get('include_initial_location', False) and initial_location
            
            if use_naming_convention:
                # libero_spatial style: pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
                grasp_prefix_filename = naming_convention.get('grasp_prefix_filename', operators_config['can_grasp']['prefix_filename_lang'])
                grasp_prefix_bddl = naming_convention.get('grasp_prefix_bddl', operators_config['can_grasp']['prefix_bddl_lang'])
                place_connector_filename = naming_convention.get('place_connector_filename', 'on_the')
                place_connector_bddl = naming_convention.get('place_connector_bddl', 'on the')
                
                # Check for object name overrides in naming convention
                object_name_overrides = naming_convention.get('object_name_overrides', {})
                if obj1_name in object_name_overrides:
                    obj1_filename = object_name_overrides[obj1_name].get('filename', obj1_config['object_filename_lang'])
                    obj1_bddl_override = object_name_overrides[obj1_name].get('bddl', obj1_config['object_bddl_lang'])
                else:
                    obj1_filename = obj1_config['object_filename_lang']
                    obj1_bddl_override = obj1_config['object_bddl_lang']
                
                obj2_filename = obj2_config['object_filename_lang']
                
                # Determine connector and prefix for placement location (obj2)
                # Check naming_convention for placement_prefix_overrides first
                obj2_name = task_spec['obj2']
                placement_prefix_overrides = naming_convention.get('placement_prefix_overrides', {})
                
                if obj2_name in placement_prefix_overrides:
                    # Use the override connector from naming_convention
                    override = placement_prefix_overrides[obj2_name]
                    base_connector_filename = override.get('place_connector_filename', place_connector_filename.replace('_on_the', ''))
                    base_connector_bddl = override.get('place_connector_bddl', place_connector_bddl.replace(' on the', ''))
                    
                    # Get prefix from object's can_place_on operator
                    placement_prefix_filename = 'on_the'  # default
                    placement_prefix_bddl = 'on the'  # default
                    obj2_ops = obj2_config.get('operators', {})
                    if isinstance(obj2_ops, dict) and 'can_place_on' in obj2_ops:
                        place_on_op = obj2_ops['can_place_on']
                        if isinstance(place_on_op, dict):
                            placement_prefix_filename = place_on_op.get('prefix_filename_lang_override', 'on_the')
                            placement_prefix_bddl = place_on_op.get('prefix_bddl_lang_override', 'on the')
                else:
                    # Use default connector from naming_convention
                    base_connector_filename = place_connector_filename.replace('_on_the', '')
                    base_connector_bddl = place_connector_bddl.replace(' on the', '')
                    
                    # Get prefix from object's can_place_on operator (check for override)
                    placement_prefix_filename = 'on_the'  # default
                    placement_prefix_bddl = 'on the'  # default
                    obj2_ops = obj2_config.get('operators', {})
                    if isinstance(obj2_ops, dict) and 'can_place_on' in obj2_ops:
                        place_on_op = obj2_ops['can_place_on']
                        if isinstance(place_on_op, dict):
                            placement_prefix_filename = place_on_op.get('prefix_filename_lang_override', 'on_the')
                            placement_prefix_bddl = place_on_op.get('prefix_bddl_lang_override', 'on the')
                
                # Combine connector and prefix
                final_place_connector_filename = f"{base_connector_filename}_{placement_prefix_filename}"
                final_place_connector_bddl = f"{base_connector_bddl} {placement_prefix_bddl}"
                
                # Get initial location name (match by base name if needed)
                initial_location_config = get_object_config(initial_location, objects_config)
                if initial_location_config:
                    initial_location_filename = initial_location_config['object_filename_lang']
                    initial_location_bddl = initial_location_config['object_bddl_lang']
                    
                    # Determine prefix for initial location based on relation type (On/In)
                    if initial_location_relation == 'In':
                        # Object is inside a container
                        initial_prefix_filename = "in_the"
                        initial_prefix_bddl = "in the"
                    else:
                        # Object is on a surface (On relation) - check for initial location prefix override first
                        initial_ops = initial_location_config.get('operators', {})
                        if isinstance(initial_ops, dict) and 'can_place_on' in initial_ops:
                            place_on_op = initial_ops['can_place_on']
                            if isinstance(place_on_op, dict):
                                # Check for initial_location_prefix override first (for objects like table_center that use "from" when picked from)
                                initial_prefix_filename = place_on_op.get('initial_location_prefix_filename_lang_override')
                                initial_prefix_bddl = place_on_op.get('initial_location_prefix_bddl_lang_override')
                                # Fall back to regular prefix override if initial_location prefix not specified
                                if initial_prefix_filename is None:
                                    initial_prefix_filename = place_on_op.get('prefix_filename_lang_override', 'on_the')
                                if initial_prefix_bddl is None:
                                    initial_prefix_bddl = place_on_op.get('prefix_bddl_lang_override', 'on the')
                            else:
                                initial_prefix_filename = 'on_the'
                                initial_prefix_bddl = 'on the'
                        else:
                            initial_prefix_filename = 'on_the'
                            initial_prefix_bddl = 'on the'
                else:
                    # Region not in config - this should not happen if all locations are properly defined in YAML
                    # Try to find it with different name variations as a fallback
                    initial_location_config = None
                    
                    # Try to find by base name match first
                    initial_location_config = get_object_config(initial_location, objects_config)
                    
                    if initial_location_config is None:
                        # Try to find by matching object_filename_lang or object_bddl_lang
                        for obj_name, obj_config in objects_config.items():
                            if obj_config.get('object_filename_lang') == initial_location or \
                               obj_config.get('object_bddl_lang') == initial_location:
                                initial_location_config = obj_config
                                break
                    
                    if initial_location_config:
                        # Found it, use the config
                        initial_location_filename = initial_location_config.get('object_filename_lang', initial_location)
                        initial_location_bddl = initial_location_config.get('object_bddl_lang', initial_location.replace('_', ' '))
                        
                        # Determine prefix based on relation type and config
                        if initial_location_relation == 'In':
                            initial_prefix_filename = 'in_the'
                            initial_prefix_bddl = 'in the'
                        else:
                            # Check for initial location prefix override in can_place_on first
                            initial_ops = initial_location_config.get('operators', {})
                            if isinstance(initial_ops, dict) and 'can_place_on' in initial_ops:
                                place_on_op = initial_ops['can_place_on']
                                if isinstance(place_on_op, dict):
                                    # Check for initial_location_prefix override first (for objects like table_center that use "from" when picked from)
                                    initial_prefix_filename = place_on_op.get('initial_location_prefix_filename_lang_override')
                                    initial_prefix_bddl = place_on_op.get('initial_location_prefix_bddl_lang_override')
                                    # Fall back to regular prefix override if initial_location prefix not specified
                                    if initial_prefix_filename is None:
                                        initial_prefix_filename = place_on_op.get('prefix_filename_lang_override', 'on_the')
                                    if initial_prefix_bddl is None:
                                        initial_prefix_bddl = place_on_op.get('prefix_bddl_lang_override', 'on the')
                                else:
                                    initial_prefix_filename = 'on_the'
                                    initial_prefix_bddl = 'on the'
                            else:
                                initial_prefix_filename = 'on_the'
                                initial_prefix_bddl = 'on the'
                    else:
                        # Location not found in config - raise error instead of inferring
                        raise ValueError(
                            f"Initial location '{initial_location}' not found in objects_config. "
                            f"Please add it to tasks_spec.yaml with appropriate object_filename_lang and object_bddl_lang. "
                            f"This location was found in the base BDDL's (:init) section but is not configured in the YAML."
                        )
                
                # Check if we should include target object's initial location in the name
                target_location_suffix_filename = ""
                target_location_suffix_bddl = ""
                include_target_initial_location_for_objects = naming_convention.get('include_target_initial_location_for_objects', [])
                if include_target_initial_location_for_objects and target_initial_location:
                    # Get base name of obj2 (e.g., akita_black_bowl from akita_black_bowl_1)
                    obj2_base_name = get_base_object_name(obj2_name)
                    if obj2_base_name in include_target_initial_location_for_objects:
                        # Get target's initial location config
                        target_location_config = get_object_config(target_initial_location, objects_config)
                        if target_location_config:
                            target_location_filename = target_location_config['object_filename_lang']
                            target_location_bddl = target_location_config['object_bddl_lang']
                            
                            # Determine prefix for target's initial location
                            if target_initial_location_relation == 'In':
                                target_location_prefix_filename = "in_the"
                                target_location_prefix_bddl = "in the"
                            else:
                                # Check for prefix override in target location's config
                                target_location_ops = target_location_config.get('operators', {})
                                if isinstance(target_location_ops, dict) and 'can_place_on' in target_location_ops:
                                    place_on_op = target_location_ops['can_place_on']
                                    if isinstance(place_on_op, dict):
                                        target_location_prefix_filename = place_on_op.get('prefix_filename_lang_override', 'on_the')
                                        target_location_prefix_bddl = place_on_op.get('prefix_bddl_lang_override', 'on the')
                                    else:
                                        target_location_prefix_filename = 'on_the'
                                        target_location_prefix_bddl = 'on the'
                                else:
                                    target_location_prefix_filename = 'on_the'
                                    target_location_prefix_bddl = 'on the'
                            
                            # Add target location suffix to filename and BDDL language
                            target_location_suffix_filename = f"_{target_location_prefix_filename}_{target_location_filename}"
                            target_location_suffix_bddl = f" {target_location_prefix_bddl} {target_location_bddl}"
                
                # Build filename: pick_up_the_{obj1}_on_the_{initial_location}_and_place_it_{placement_prefix}_{obj2}[_{target_location}]
                filename = f"{grasp_prefix_filename}_{obj1_filename}_{initial_prefix_filename}_{initial_location_filename}_{final_place_connector_filename}_{obj2_filename}{target_location_suffix_filename}"
                
                # Build BDDL language: Pick the {obj1} on the {initial_location} and place it {placement_prefix} {obj2}[ {target_location}]
                obj2_bddl = obj2_config['object_bddl_lang']
                bddl_lang = f"{grasp_prefix_bddl} {obj1_bddl_override} {initial_prefix_bddl} {initial_location_bddl} {final_place_connector_bddl} {obj2_bddl}{target_location_suffix_bddl}"
                bddl_lang = bddl_lang.lower()
            else:
                # Default libero_goal style: put_the_bowl_on_the_plate
                obj1_prefix = operators_config['can_grasp']['prefix_filename_lang']
                obj1_filename = obj1_config['object_filename_lang']
                
                # Check for prefix override for obj2 (the place target) in the object's operator config
                place_on_config = operators_config['can_place_on']
                obj2_operator_config = obj2_config.get('operators', {})
                if isinstance(obj2_operator_config, dict):
                    obj2_can_place_on_config = obj2_operator_config.get('can_place_on', {})
                    obj2_prefix = obj2_can_place_on_config.get('prefix_filename_lang_override')
                else:
                    obj2_prefix = None
                
                if obj2_prefix is None:
                    obj2_prefix = place_on_config['prefix_filename_lang']
                obj2_filename = obj2_config['object_filename_lang']
                filename = f"{obj1_prefix}_{obj1_filename}_{obj2_prefix}_{obj2_filename}"
                
                obj1_prefix_bddl = operators_config['can_grasp']['prefix_bddl_lang']
                obj1_bddl = obj1_config['object_bddl_lang']
                # Check for BDDL language override for obj2 in the object's operator config
                if isinstance(obj2_operator_config, dict):
                    obj2_can_place_on_config = obj2_operator_config.get('can_place_on', {})
                    obj2_prefix_bddl = obj2_can_place_on_config.get('prefix_bddl_lang_override')
                else:
                    obj2_prefix_bddl = None
                
                if obj2_prefix_bddl is None:
                    obj2_prefix_bddl = place_on_config['prefix_bddl_lang']
                obj2_bddl = obj2_config['object_bddl_lang']
                bddl_lang = f"{obj1_prefix_bddl} {obj1_bddl} {obj2_prefix_bddl} {obj2_bddl}"
                bddl_lang = bddl_lang.lower()
    elif task_type == 'In':
        # open_the_top_drawer_and_put_the_bowl_inside
        obj1_name = task_spec['obj1']  # container
        obj2_name = task_spec['obj2']  # object to place

        # Get configs, matching by base name if needed
        obj1_config = get_object_config(obj1_name, objects_config)
        obj2_config = get_object_config(obj2_name, objects_config)

        if obj1_config is None or obj2_config is None:
            # Provide helpful error message showing both instance names and base names we tried to match
            obj1_base = get_base_object_name(obj1_name)
            obj2_base = get_base_object_name(obj2_name)

            missing_objs = []
            if obj1_config is None:
                missing_objs.append(f"{obj1_name} (tried base: {obj1_base})")
            if obj2_config is None:
                missing_objs.append(f"{obj2_name} (tried base: {obj2_base})")

            available_keys = list(objects_config.keys())[:10]  # Show first 10 for debugging
            raise ValueError(
                f"Objects not in config: {', '.join(missing_objs)}. "
                f"Available config keys (first 10): {available_keys}"
            )

        obj1_prefix = operators_config['can_open']['prefix_filename_lang']
        obj1_filename = obj1_config['object_filename_lang']
        obj2_prefix = operators_config['can_grasp']['prefix_filename_lang']
        obj2_filename = obj2_config['object_filename_lang']
        filename = f"{obj1_prefix}_{obj1_filename}_and_{obj2_prefix}_{obj2_filename}_inside"

        obj1_prefix_bddl = operators_config['can_open']['prefix_bddl_lang']
        obj1_bddl = obj1_config['object_bddl_lang']
        obj2_prefix_bddl = operators_config['can_grasp']['prefix_bddl_lang']
        obj2_bddl = obj2_config['object_bddl_lang']
        bddl_lang = f"{obj1_prefix_bddl} {obj1_bddl} and {obj2_prefix_bddl} {obj2_bddl} inside"
    elif task_type == 'PlaceIn':
        # put_the_bowl_in_the_top_drawer (place into already-open drawer; no open step)
        obj1_name = task_spec['obj1']  # container
        obj2_name = task_spec['obj2']  # object to place

        obj1_config = get_object_config(obj1_name, objects_config)
        obj2_config = get_object_config(obj2_name, objects_config)

        if obj1_config is None or obj2_config is None:
            obj1_base = get_base_object_name(obj1_name)
            obj2_base = get_base_object_name(obj2_name)
            missing_objs = []
            if obj1_config is None:
                missing_objs.append(f"{obj1_name} (tried base: {obj1_base})")
            if obj2_config is None:
                missing_objs.append(f"{obj2_name} (tried base: {obj2_base})")
            available_keys = list(objects_config.keys())[:10]
            raise ValueError(
                f"Objects not in config: {', '.join(missing_objs)}. "
                f"Available config keys (first 10): {available_keys}"
            )

        obj2_prefix = operators_config['can_grasp']['prefix_filename_lang']  # put_the
        obj2_filename = obj2_config['object_filename_lang']
        obj1_filename = obj1_config['object_filename_lang']
        filename = f"{obj2_prefix}_{obj2_filename}_in_the_{obj1_filename}"

        obj2_prefix_bddl = operators_config['can_grasp']['prefix_bddl_lang']  # Put the
        obj2_bddl = obj2_config['object_bddl_lang']
        obj1_bddl = obj1_config['object_bddl_lang']
        bddl_lang = f"{obj2_prefix_bddl} {obj2_bddl} in the {obj1_bddl}"
    elif task_type == 'Open':
        obj1_name = task_spec['obj1']
        
        # Get config, matching by base name if needed
        obj1_config = get_object_config(obj1_name, objects_config)
        
        if obj1_config is None:
            obj1_base = get_base_object_name(obj1_name)
            available_keys = list(objects_config.keys())[:10]  # Show first 10 for debugging
            raise ValueError(
                f"Object not in config: {obj1_name} (tried base: {obj1_base}). "
                f"Available config keys (first 10): {available_keys}"
            )
        
        obj1_prefix = operators_config['can_open']['prefix_filename_lang']
        obj1_filename = obj1_config['object_filename_lang']
        filename = f"{obj1_prefix}_{obj1_filename}"
        
        obj1_prefix_bddl = operators_config['can_open']['prefix_bddl_lang']
        obj1_bddl = obj1_config['object_bddl_lang']
        bddl_lang = f"{obj1_prefix_bddl} {obj1_bddl}"
    elif task_type == 'Turnon':
        obj1_name = task_spec['obj1']
        
        # Get config, matching by base name if needed
        obj1_config = get_object_config(obj1_name, objects_config)
        
        if obj1_config is None:
            obj1_base = get_base_object_name(obj1_name)
            available_keys = list(objects_config.keys())[:10]  # Show first 10 for debugging
            raise ValueError(
                f"Object not in config: {obj1_name} (tried base: {obj1_base}). "
                f"Available config keys (first 10): {available_keys}"
            )
        
        obj1_prefix = operators_config['can_turn_on']['prefix_filename_lang']
        obj1_filename = obj1_config['object_filename_lang']
        filename = f"{obj1_prefix}_{obj1_filename}"
        
        obj1_prefix_bddl = operators_config['can_turn_on']['prefix_bddl_lang']
        obj1_bddl = obj1_config['object_bddl_lang']
        bddl_lang = f"{obj1_prefix_bddl} {obj1_bddl}"
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return filename, bddl_lang


def generate_execution_steps(
    task_spec: Dict,
    objects_config: Dict,
    operators_config: Dict
) -> List[str]:
    """Generate execution steps for a task specification"""
    task_type = task_spec['type']
    steps = []
    
    if task_type == 'On':
        variant = task_spec['variant']
        obj1_name = task_spec['obj1']
        obj2_name = task_spec['obj2']
        
        if variant == 'push':
            steps.append(f"Touch {obj1_name}")
            steps.append(f"Place {obj1_name} {obj2_name}")
        else:
            steps.append(f"Grasp {obj1_name}")
            steps.append(f"Place {obj1_name} {obj2_name}")
    elif task_type == 'In':
        obj1_name = task_spec['obj1']  # container
        obj2_name = task_spec['obj2']  # object to place
        steps.append(f"Open {obj1_name}")
        steps.append(f"Grasp {obj2_name}")
        steps.append(f"Place {obj2_name} {obj1_name}")
    elif task_type == 'PlaceIn':
        obj1_name = task_spec['obj1']  # container (already open)
        obj2_name = task_spec['obj2']  # object to place
        steps.append(f"Grasp {obj2_name}")
        steps.append(f"Place {obj2_name} {obj1_name}")
    elif task_type == 'Open':
        obj1_name = task_spec['obj1']
        steps.append(f"Open {obj1_name}")
    elif task_type == 'Turnon':
        obj1_name = task_spec['obj1']
        steps.append(f"Turnon {obj1_name}")
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return steps


def generate_views_only(
    extra_envs_config_path: str,
    splits_to_process: Optional[List[str]] = None,
    suffix: Optional[str] = None
) -> None:
    """
    Generate views from existing BDDL files and metadata without regenerating BDDLs or init states.
    
    Args:
        extra_envs_config_path: Path to tasks_spec.yaml
        splits_to_process: Optional list of splits to process (if None, processes all splits)
        suffix: Optional suffix for paths
    """
    with open(extra_envs_config_path, "r", encoding="utf-8") as f:
        extra_envs_config = yaml.safe_load(f)
    
    # Determine base paths
    original_bddl_files_base = get_libero_path('bddl_files')
    bddl_files_base = original_bddl_files_base
    if suffix is not None:
        suffix_with_underscore = suffix if suffix.startswith('_') else '_' + suffix
        bddl_files_base = original_bddl_files_base + suffix_with_underscore
    
    original_init_states_base = get_libero_path('init_states')
    init_states_base = original_init_states_base
    if suffix is not None:
        suffix_with_underscore = suffix if suffix.startswith('_') else '_' + suffix
        init_states_base = original_init_states_base + suffix_with_underscore
    
    # Filter splits if splits_to_process is specified
    splits_to_iterate = extra_envs_config["splits"]
    if splits_to_process is not None:
        splits_to_iterate = {
            split_name: split_config
            for split_name, split_config in splits_to_iterate.items()
            if split_name in splits_to_process
        }
        if not splits_to_iterate:
            print(f"WARNING: No splits found matching {splits_to_process}. Available splits: {list(extra_envs_config['splits'].keys())}")
            return
        print(f"Processing views for splits: {list(splits_to_iterate.keys())}")
    
    all_split_view_matching_tasks = {}
    
    for split_name, split_config in splits_to_iterate.items():
        views_config = split_config.get('views', {})
        if not views_config:
            print(f"No views configured for split '{split_name}', skipping")
            continue
        
        print(f"\nProcessing views for split '{split_name}'...")
        
        # Load metadata from existing task_metadata.yaml files
        task_metadata_by_dir, bddl_dirs = load_metadata_from_files(
            split_name,
            bddl_files_base,
            suffix,
            views_config  # Pass views_config to exclude view directories
        )
        
        if not task_metadata_by_dir:
            print(f"WARNING: No metadata found for split '{split_name}', skipping views")
            continue
        
        # Process views
        # For generate_views_only, we always copy init states, so gen_init_states=True
        view_matching_tasks = process_views_for_split_bddl(
            split_name,
            views_config,
            task_metadata_by_dir,
            bddl_dirs,
            bddl_files_base,
            init_states_base,
            suffix,
            gen_init_states=True
        )
        
        all_split_view_matching_tasks[split_name] = {
            'view_matching_tasks': view_matching_tasks,
            'views_config': views_config
        }
    
    # Copy init states for views
    if all_split_view_matching_tasks:
        print("\n" + "="*80)
        print("COPYING INIT STATES FOR VIEWS")
        print("="*80)
        for split_name, view_data in all_split_view_matching_tasks.items():
            view_matching_tasks = view_data['view_matching_tasks']
            views_config = view_data['views_config']
            
            # Build init_states_dirs dict
            # For existing tasks, use default (non-suffixed) path if suffix is provided
            # because existing tasks' init states are not copied to suffixed directories
            default_init_states_base = get_libero_path('init_states')
            if suffix is not None:
                init_states_dirs = {
                    'existing': os.path.join(default_init_states_base, split_name),
                    'extra': os.path.join(init_states_base, f'{split_name}_extra')
                }
            else:
                init_states_dirs = {
                    'existing': os.path.join(init_states_base, split_name),
                    'extra': os.path.join(init_states_base, f'{split_name}_extra')
                }
            
            # For generate_views_only, we always copy init states
            copy_init_states_for_views(
                split_name,
                views_config,
                view_matching_tasks,
                init_states_dirs,
                init_states_base,
                suffix,
                gen_init_states=True
            )
        print("="*80 + "\n")
    else:
        print("\nNo views to process.")


def gen_extra_libero_envs(extra_envs_config_path: str, overwrite_init_states: bool, no_overwrite_init_states: bool, gen_init_states: bool, num_workers_init_states: int = 10, splits_to_process: Optional[List[str]] = None, only_tasks: Optional[List[str]] = None, suffix: Optional[str] = None, min_region_size: float = 0.0064):
    with open(extra_envs_config_path, "r", encoding="utf-8") as f:
        extra_envs_config = yaml.safe_load(f)

    objects_config = extra_envs_config["objects"]
    operators_config = extra_envs_config["operators"]
    
    # Collect individual BDDL files that need init states generated (after all YAMLs are generated)
    init_state_tasks: List[Tuple[str, str, bool]] = []
    
    # Filter splits if splits_to_process is specified
    splits_to_iterate = extra_envs_config["splits"]
    if splits_to_process is not None:
        splits_to_iterate = {
            split_name: split_config
            for split_name, split_config in splits_to_iterate.items()
            if split_name in splits_to_process
        }
        if not splits_to_iterate:
            print(f"WARNING: No splits found matching {splits_to_process}. Available splits: {list(extra_envs_config['splits'].keys())}")
            return
        print(f"Processing splits: {list(splits_to_iterate.keys())}")

    # If suffix is provided, copy base split .bddl files to suffixed directory at the start
    # Then we'll work entirely in the suffixed directory
    original_bddl_files_base = get_libero_path('bddl_files')
    bddl_files_base = original_bddl_files_base
    if suffix is not None:
        # Add underscore before suffix if it doesn't already start with one
        suffix_with_underscore = suffix if suffix.startswith('_') else '_' + suffix
        bddl_files_base = original_bddl_files_base + suffix_with_underscore
        
        print("\n" + "="*80)
        print("COPYING BASE SPLIT .BDDL FILES TO SUFFIXED DIRECTORY")
        print("="*80)
        
        # Copy base split .bddl files for each split we'll process
        for split_name in splits_to_iterate.keys():
            source_split_dir = os.path.join(original_bddl_files_base, split_name)
            dest_split_dir = os.path.join(bddl_files_base, split_name)
            
            if not os.path.exists(source_split_dir):
                print(f"WARNING: Source split directory does not exist: {source_split_dir}")
                continue
            
            # Create destination directory
            Path(dest_split_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy all .bddl files from source to destination
            bddl_files = list(Path(source_split_dir).glob("*.bddl"))
            copied_count = 0
            for bddl_file in bddl_files:
                dest_file = os.path.join(dest_split_dir, bddl_file.name)
                shutil.copy2(bddl_file, dest_file)
                copied_count += 1
            
            print(f"Copied {copied_count} .bddl files from {split_name} to suffixed directory")
        
        print("="*80 + "\n")
    
    # Determine overwrite_init_states value based on flags and user input
    if gen_init_states:
        # If both flags are provided, this should have been caught by argument parser, but check anyway
        if overwrite_init_states and no_overwrite_init_states:
            raise ValueError("Cannot specify both --overwrite-init-states and --no-overwrite-init-states")
        
        # Determine the base init states directory
        init_states_base = get_libero_path('init_states')
        init_states_output_base = init_states_base
        if suffix is not None:
            suffix_with_underscore = suffix if suffix.startswith('_') else '_' + suffix
            init_states_output_base = init_states_base + suffix_with_underscore
        
        # Check if existing init states exist for the splits being processed
        existing_init_states_found = False
        for split_name in splits_to_iterate.keys():
            out_folder_name_extra = f'{split_name}_extra'
            out_init_states_dir_extra = os.path.join(init_states_output_base, out_folder_name_extra)
            
            # Check if init state directories exist and contain files
            if os.path.exists(out_init_states_dir_extra) and os.listdir(out_init_states_dir_extra):
                existing_init_states_found = True
                break
        
        # Determine overwrite_init_states value
        if overwrite_init_states:
            # Explicitly set to overwrite
            overwrite_init_states = True
        elif no_overwrite_init_states:
            # Explicitly set to not overwrite
            overwrite_init_states = False
        elif existing_init_states_found:
            # Neither flag provided and init states exist - prompt user
            print("\n" + "="*80)
            print("INIT STATES OVERWRITE PROMPT")
            print("="*80)
            print("Existing init states found for the splits being processed.")
            response = input("Do you want to overwrite existing init states? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                overwrite_init_states = True
                print("Will overwrite existing init states.")
            else:
                overwrite_init_states = False
                print("Will skip existing init states (will not overwrite).")
            print("="*80 + "\n")
        else:
            # Neither flag provided and no init states exist - default to False (won't overwrite since nothing exists)
            overwrite_init_states = False
    
    # Track task counts per split
    split_task_counts = {}
    # Track view matching tasks per split (for later init state copying)
    all_split_view_matching_tasks = {}

    for split_name, split_config in splits_to_iterate.items():
        # Get duplicate_mode (can be 'true', 'false', or 'only_relevant_objects', defaults to 'true')
        # Support backward compatibility with allow_duplicate_tasks
        duplicate_mode = split_config.get('duplicate_mode', None)
        if duplicate_mode is None:
            # Check for old allow_duplicate_tasks flag
            allow_duplicate_tasks_old = split_config.get('allow_duplicate_tasks', True)
            duplicate_mode = 'true' if allow_duplicate_tasks_old else 'false'
        
        # Normalize duplicate_mode to handle boolean values from YAML
        if duplicate_mode is True or duplicate_mode == 'true':
            duplicate_mode = 'true'
        elif duplicate_mode is False or duplicate_mode == 'false':
            duplicate_mode = 'false'
        elif duplicate_mode == 'only_relevant_objects':
            duplicate_mode = 'only_relevant_objects'
        else:
            # Default to 'true' if invalid value
            print(f"WARNING: Invalid duplicate_mode '{duplicate_mode}' for split '{split_name}'. Defaulting to 'true'.")
            duplicate_mode = 'true'
        
        # Apply object overrides for this split (e.g., graspable_objects)
        object_overrides = split_config.get('object_overrides', {})
        split_objects_config = apply_object_overrides(objects_config, object_overrides)
        
        task_metadata_extra = {}
        task_metadata_existing = {}
        out_folder_name_extra = f'{split_name}_extra'
        # bddl_files_base is now set to suffixed directory if suffix was provided, otherwise original
        # All operations will use this base (which may be suffixed)
        init_states_base = get_libero_path('init_states')
        init_states_output_base = init_states_base
        if suffix is not None:
            # Add underscore before suffix if it doesn't already start with one
            suffix_with_underscore = suffix if suffix.startswith('_') else '_' + suffix
            init_states_output_base = init_states_base + suffix_with_underscore
        out_bddl_dir_extra = os.path.join(bddl_files_base, out_folder_name_extra)
        out_bddl_dir_original = os.path.join(bddl_files_base, split_name)  # Base split directory (now in suffixed location if suffix provided)
        out_init_states_dir_extra = os.path.join(init_states_output_base, out_folder_name_extra)
        
        # Convert only_tasks to a set for efficient lookup
        only_tasks_set = None
        if only_tasks is not None:
            only_tasks_set = set(only_tasks)
            print(f"Only generating tasks: {only_tasks_set}")
        
        # Only delete the BDDL directories if not using --only-tasks flag
        # When using --only-tasks, we'll overwrite existing files instead
        if only_tasks_set is None:
            # Always delete the BDDL directories before generating new tasks
            if os.path.exists(out_bddl_dir_extra):
                shutil.rmtree(out_bddl_dir_extra)
        
        existing_tasks_filenames_all = os.listdir(out_bddl_dir_original)
        existing_tasks_filenames_all = set([x.replace('.bddl', '') for x in existing_tasks_filenames_all if x.endswith('.bddl')])
        existing_tasks_filename_matches = set()
        # Track all newly generated task names (including with suffixes) to avoid conflicts within the same run
        generated_task_names = set()
        # Track BDDL content and execution steps of generated tasks to check for functional equivalence
        # Format: {filename: {'bddl_content': str, 'execution_steps': List[str], 'is_chained': bool}}
        generated_task_info = {}
        
        # Get invalid_base_tasks if specified (tasks to exclude)
        invalid_base_tasks = split_config.get('invalid_base_tasks', [])
        invalid_base_tasks_set = set(invalid_base_tasks)
        
        # Get skip_generation_for conditions if specified
        skip_generation_for = split_config.get('skip_generation_for', [])
        
        # Automatically load existing tasks metadata from BDDL files (excluding invalid tasks)
        # Also collect all objects/regions that are present in existing BDDL files
        existing_tasks_metadata = {}
        objects_in_existing_bddls = set()
        if os.path.exists(out_bddl_dir_original):
            for bddl_filename in os.listdir(out_bddl_dir_original):
                if not bddl_filename.endswith('.bddl'):
                    continue
                
                task_name = bddl_filename.replace('.bddl', '')
                
                bddl_path = os.path.join(out_bddl_dir_original, bddl_filename)
                
                try:
                    with open(bddl_path, "r", encoding="utf-8") as f:
                        bddl_content = f.read()
                    
                    parser = BDDLParser(bddl_content)
                    goal_conditions = parser.goal_conditions
                    
                    # Collect all objects/regions from this BDDL file
                    # all_objects includes both (:objects) and (:obj_of_interest)
                    objects_in_existing_bddls.update(parser.all_objects)
                    
                    # Also collect objects/regions from (:init) section (locations where objects are placed)
                    # These are the second arguments in (On obj location) and (In obj location) predicates
                    # This ensures regions like wooden_cabinet_1_top_region are included even if not in (:obj_of_interest)
                    init_states = parser.initial_states
                    for location in init_states.values():
                        objects_in_existing_bddls.add(location)
                    
                    if not goal_conditions:
                        print(f"WARNING: Could not parse goal conditions from {bddl_filename}, skipping")
                        continue
                    
                    # Infer execution steps from goal conditions
                    # Use base objects_config for existing tasks (they were created with base config)
                    execution_steps = infer_execution_steps_from_goal(
                        goal_conditions, objects_config, operators_config
                    )
                    
                    # Get initial location of the first action's object for matching purposes
                    # This is needed to ensure we only match tasks where objects start at the same location
                    initial_location_info = None
                    if execution_steps:
                        first_step = execution_steps[0]
                        stage, obj = parse_execution_step(first_step)
                        
                        # For In tasks, the first step is 'Open container', but we need the initial location
                        # of the object being placed (which is in the second step 'Grasp obj2')
                        if stage == 'Open' and len(execution_steps) > 1:
                            # Check if second step is Grasp (indicating this is an In task)
                            second_step = execution_steps[1]
                            second_stage, second_obj = parse_execution_step(second_step)
                            if second_stage == 'Grasp':
                                # This is an In task - get initial location of the object being placed
                                result = parser.get_initial_location_relation(second_obj)
                                if result is not None:
                                    init_loc, init_rel = result
                                    initial_location_info = {
                                        'object': second_obj,
                                        'initial_location': init_loc,
                                        'initial_location_relation': init_rel
                                    }
                        elif stage in ['Grasp', 'Open', 'Turnon']:
                            # For Grasp, Open (standalone), and Turnon tasks, get initial location of the object
                            result = parser.get_initial_location_relation(obj)
                            if result is not None:
                                init_loc, init_rel = result
                                initial_location_info = {
                                    'object': obj,
                                    'initial_location': init_loc,
                                    'initial_location_relation': init_rel
                                }
                    
                    existing_tasks_metadata[task_name] = {
                        'execution_steps': execution_steps,
                        'initial_states': parser.initial_states,  # Store full init state for matching
                        'initial_location_info': initial_location_info,
                        'is_existing_task': True
                    }
                except Exception as e:
                    print(f"WARNING: Error parsing {bddl_filename}: {e}, skipping")
                    continue
        
        # Filter split_objects_config to only include objects/regions present in existing BDDL files
        # This ensures each split only uses objects/regions that actually exist in its BDDL files
        # Match by base name: objects_in_existing_bddls may have instances (_1, _2), while
        # split_objects_config may have base names
        if objects_in_existing_bddls:
            # Build set of base names from objects_in_existing_bddls
            existing_bddls_base_names = {get_base_object_name(obj) for obj in objects_in_existing_bddls}
            # Also include exact matches for objects that don't have numeric suffixes
            existing_bddls_base_names.update(objects_in_existing_bddls)
            
            # Only keep objects/regions that match (by base name or exact match) objects in existing BDDL files
            filtered_split_objects_config = {
                obj_name: obj_config
                for obj_name, obj_config in split_objects_config.items()
                if obj_name in existing_bddls_base_names or any(
                    get_base_object_name(existing_obj) == get_base_object_name(obj_name)
                    for existing_obj in objects_in_existing_bddls
                )
            }
            split_objects_config = filtered_split_objects_config
            print(f"Filtered objects_config for {split_name}: {len(split_objects_config)} objects/regions (from {len(objects_in_existing_bddls)} found in BDDL files)")
        
        # All existing tasks should be generated/validated, regardless of invalid_base_tasks
        # invalid_base_tasks only affects which tasks can be used as base tasks for generating new tasks
        existing_tasks_filenames = existing_tasks_filenames_all
        
        # Get default grasp strategy (object -> default task for grasping that object)
        default_grasp_strategy = split_config.get('default_grasp_strategy', {})
        
        # Validate default_grasp_strategy references existing tasks
        for obj_name, task_name in default_grasp_strategy.items():
            if task_name not in existing_tasks_metadata:
                raise ValueError(f"default_grasp_strategy for '{obj_name}' references task '{task_name}' which is not in existing tasks")
        
        # Get default actions_from strategy (first_step -> default task for that first step)
        # e.g., "Grasp wine_bottle_1": "put_the_wine_bottle_on_top_of_the_cabinet"
        default_actions_from_strategy = split_config.get('default_actions_from_strategy', {})
        
        # Validate default_actions_from_strategy references existing tasks
        for first_step, task_name in default_actions_from_strategy.items():
            if task_name not in existing_tasks_metadata:
                raise ValueError(f"default_actions_from_strategy for '{first_step}' references task '{task_name}' which is not in existing tasks")
        
        # Get task-specific overrides (task_name -> {property: value})
        task_overrides = split_config.get('task_overrides', {})
        
        # Validate task_overrides grasps_from references existing tasks
        for task_name, overrides in task_overrides.items():
            if 'grasps_from' in overrides:
                grasps_from_task = overrides['grasps_from']
                if grasps_from_task not in existing_tasks_metadata:
                    raise ValueError(f"task_overrides['{task_name}']['grasps_from'] references task '{grasps_from_task}' which is not in existing tasks")
        
        # Get pattern-based task overrides
        task_pattern_overrides = split_config.get('task_pattern_overrides', [])
        
        # Validate task_pattern_overrides
        for pattern_override in task_pattern_overrides:
            pattern = pattern_override.get('pattern', {})
            overrides = pattern_override.get('overrides', {})
            
            # Validate grasps_from if specified
            if 'grasps_from' in overrides:
                grasps_from = overrides['grasps_from']
                if isinstance(grasps_from, dict):
                    for obj_name, grasp_info in grasps_from.items():
                        if isinstance(grasp_info, dict) and 'task' in grasp_info:
                            task_name = grasp_info['task']
                            if task_name not in existing_tasks_metadata:
                                raise ValueError(f"task_pattern_overrides grasps_from for '{obj_name}' references task '{task_name}' which is not in existing tasks")
        
        # Get valid base tasks
        valid_base_tasks = split_config.get('valid_base_tasks', [])
        
        # Determine valid_base_tasks based on what's specified:
        # 1. If valid_base_tasks is specified, use it (and filter out invalid_base_tasks if specified)
        # 2. If invalid_base_tasks is specified but not valid_base_tasks, use all existing tasks minus invalid ones
        # 3. If neither is specified, use all existing tasks
        if valid_base_tasks:
            # If both are specified, remove invalid ones from valid_base_tasks
            if invalid_base_tasks:
                valid_base_tasks = [task for task in valid_base_tasks if task not in invalid_base_tasks_set]
        elif invalid_base_tasks:
            # Only invalid_base_tasks is specified, use all existing tasks minus invalid ones
            valid_base_tasks = list(existing_tasks_filenames_all - invalid_base_tasks_set)
        else:
            # Neither is specified, use all existing tasks
            valid_base_tasks = list(existing_tasks_filenames_all)
        
        # If still empty, raise an error (should only happen if there are no existing tasks)
        if not valid_base_tasks:
            raise ValueError(f"No valid base tasks found for split '{split_name}'. Check that there are existing BDDL files in the split directory.")
        
        # Sort valid_base_tasks to ensure deterministic processing order
        valid_base_tasks = sorted(valid_base_tasks)
        
        # Dictionary to store configurations with 'id' for later reference
        id_to_config = {}
        
        # Process operator definitions section - these are definitions only, no environment generation
        operator_definitions = split_config.get('new_environment_operator_definitions', [])
        for def_config in operator_definitions:
            if isinstance(def_config, dict) and 'id' in def_config:
                config_id = def_config['id']
                # Store a copy without the id field for reuse
                config_copy = {k: v for k, v in def_config.items() if k != 'id'}
                id_to_config[config_id] = config_copy
                
                # Handle nested replicate_id in definitions (for recursive references)
                if isinstance(config_copy, dict) and 'replicate_id' in config_copy:
                    nested_id = config_copy['replicate_id']
                    if nested_id in id_to_config:
                        # Merge the referenced config
                        referenced = id_to_config[nested_id].copy()
                        referenced.update({k: v for k, v in config_copy.items() if k != 'replicate_id'})
                        id_to_config[config_id] = referenced
        
        # Get new_environments configuration (for environment generation)
        new_environments = split_config.get('new_environments', [])
        
        # Helper function to process a single move operation and return new tasks
        def process_move_operation(move_config, base_bddl_content, base_task_name, split_name, bddl_files_base, objects_config):
            """Process a single move operation and return list of (task_name, bddl_content, is_swapped, swap_info) tuples."""
            from_object = move_config.get('from_object')
            to_location = move_config.get('to_location')
            from_location = move_config.get('from_location', None)
            base_tasks_filter = move_config.get('base_tasks', None)
            
            if from_object is None or to_location is None:
                print(f"Warning: Invalid move config (missing from_object or to_location): {move_config}")
                return []
            
            # If base_tasks is specified, only process moves for those base tasks
            if base_tasks_filter is not None:
                if base_task_name not in base_tasks_filter:
                    return []
            
            # Support both single location and list of locations
            # New format: to_location can contain strings or dicts where key is location and value has additional_modifications
            to_location_raw = to_location if isinstance(to_location, list) else [to_location]
            
            # Parse to_location items: extract location name and additional_modifications
            to_location_configs = []
            for item in to_location_raw:
                if isinstance(item, dict):
                    # Dictionary format: key is location, value contains additional_modifications
                    for loc_name, additional_mods in item.items():
                        to_location_configs.append({
                            'location': loc_name,
                            'additional_modifications': additional_mods
                        })
                else:
                    # String format: simple location with no additional modifications
                    to_location_configs.append({
                        'location': item,
                        'additional_modifications': {}
                    })
            
            # Find ALL object instances that match from_object (by base name)
            parser = BDDLParser(base_bddl_content)
            initial_states = parser.initial_states
            
            matching_object_instances = []
            for obj_name, location in initial_states.items():
                obj_base = get_base_object_name(obj_name)
                if obj_base == from_object:
                    # If from_location is specified, check if this object is at that location
                    if from_location is not None:
                        location_base = get_base_object_name(location)
                        from_location_base = get_base_object_name(from_location)
                        if location_base != from_location_base:
                            continue
                    matching_object_instances.append(obj_name)
            
            if not matching_object_instances:
                return []
            
            # Create a temporary perturbator
            temp_perturbator = BDDLPerturbator(
                split_name,
                base_task_name,
                bddl_files_base=bddl_files_base,
                bddl_content=base_bddl_content
            )
            
            new_tasks = []
            # Create all combinations: for each object instance, create a move to each to_location
            for object_instance in matching_object_instances:
                for to_loc_config in to_location_configs:
                    to_loc = to_loc_config['location']
                    additional_modifications = to_loc_config['additional_modifications']
                    
                    # Check if target location is already occupied
                    parser = BDDLParser(base_bddl_content)
                    initial_states = parser.initial_states
                    
                    source_object_current_location = initial_states.get(object_instance)
                    
                    # Find target location instance
                    target_location_instance = None
                    all_objects = parser.all_objects
                    for obj_name in all_objects:
                        obj_base = get_base_object_name(obj_name)
                        if obj_base == to_loc:
                            target_location_instance = obj_name
                            break
                    
                    if target_location_instance is None:
                        all_regions = parse_all_regions_from_bddl(base_bddl_content)
                        for region_name in all_regions:
                            region_base = get_base_object_name(region_name)
                            if region_base == to_loc:
                                target_location_instance = region_name
                                break
                    
                    if target_location_instance is None:
                        for location in initial_states.values():
                            location_base = get_base_object_name(location)
                            if location_base == to_loc:
                                target_location_instance = location
                                break
                    
                    # Check if target location is occupied
                    object_at_target = None
                    if target_location_instance:
                        for obj_name, location in initial_states.items():
                            if location == target_location_instance:
                                object_at_target = obj_name
                                break
                    
                    # Perform swap if target is occupied, otherwise just move
                    if object_at_target and source_object_current_location:
                        # Swap positions
                        object_at_target_base = get_base_object_name(object_at_target)
                        modified_bddl = temp_perturbator._replace_init_move(
                            base_bddl_content,
                            object_at_target_base,
                            get_base_object_name(source_object_current_location),
                            object_instance=object_at_target,
                            objects_config=objects_config
                        )
                        modified_bddl = temp_perturbator._replace_init_move(
                            modified_bddl,
                            from_object,
                            to_loc,
                            object_instance=object_instance,
                            objects_config=objects_config
                        )
                    else:
                        # Target location is empty, just move
                        modified_bddl = temp_perturbator._replace_init_move(
                            base_bddl_content,
                            from_object,
                            to_loc,
                            object_instance=object_instance,
                            objects_config=objects_config
                        )
                    
                    # Apply additional modifications if specified
                    if additional_modifications:
                        modified_bddl = temp_perturbator._apply_additional_modifications(
                            modified_bddl,
                            additional_modifications
                        )
                    
                    # Only add if actually changed something
                    if modified_bddl != base_bddl_content:
                        swap_info = {
                            'from_object': from_object,
                            'to_location': to_loc,
                            'base_task': base_task_name,
                            'object_instance': object_instance
                        }
                        swapped_task_name = f"{base_task_name}_swap_{object_instance}_to_{to_loc}"
                        
                        # Also store detailed move information for conflict detection
                        move_info = {
                            'object_instance': object_instance,
                            'from_location': source_object_current_location,
                            'to_location': target_location_instance if target_location_instance else to_loc,
                            'to_location_base': to_loc,
                            'swapped_with': object_at_target if object_at_target else None,
                            'additional_modifications': additional_modifications
                        }
                        
                        new_tasks.append((swapped_task_name, modified_bddl, True, swap_info, move_info))
            
            return new_tasks
        
        # Helper function to detect conflicts between move outcomes
        def detect_move_conflicts(move_outcomes):
            """
            Check if a combination of moves has conflicts.
            Returns (has_conflict, conflict_reason)
            """
            # Track which objects are moved
            moved_objects = set()
            # Track which locations receive objects (location -> object)
            # Use base location name for matching since different instances might refer to same location
            location_occupancy = {}
            
            for outcome in move_outcomes:
                obj_instance = outcome['object_instance']
                # Use to_location_base for conflict detection (base name, not instance)
                to_location_base = outcome.get('to_location_base', outcome.get('to_location', ''))
                
                # Check if same object is moved multiple times
                if obj_instance in moved_objects:
                    return True, f"Object {obj_instance} is moved multiple times"
                moved_objects.add(obj_instance)
                
                # Check if location already has an object (single occupancy)
                # Use base location name for comparison
                if to_location_base in location_occupancy:
                    other_obj = location_occupancy[to_location_base]
                    return True, f"Location {to_location_base} already occupied by {other_obj}, cannot place {obj_instance}"
                location_occupancy[to_location_base] = obj_instance
            
            return False, None
        
        # Helper function to apply a combination of moves to BDDL content
        def apply_move_combination(move_outcomes, base_bddl_content, base_task_name, split_name, bddl_files_base, objects_config):
            """Apply multiple moves sequentially to BDDL content."""
            temp_perturbator = BDDLPerturbator(
                split_name,
                base_task_name,
                bddl_files_base=bddl_files_base,
                bddl_content=base_bddl_content
            )
            
            current_bddl = base_bddl_content
            task_name_parts = [base_task_name]
            
            # Apply moves in order
            for outcome in move_outcomes:
                from_object_base = get_base_object_name(outcome['object_instance'])
                to_location_base = outcome['to_location_base']
                
                # Check if target location is occupied and handle swap
                parser = BDDLParser(current_bddl)
                initial_states = parser.initial_states
                source_object_current_location = initial_states.get(outcome['object_instance'])
                
                # Find target location instance
                target_location_instance = None
                all_objects = parser.all_objects
                for obj_name in all_objects:
                    obj_base = get_base_object_name(obj_name)
                    if obj_base == to_location_base:
                        target_location_instance = obj_name
                        break
                
                if target_location_instance is None:
                    all_regions = parse_all_regions_from_bddl(current_bddl)
                    for region_name in all_regions:
                        region_base = get_base_object_name(region_name)
                        if region_base == to_location_base:
                            target_location_instance = region_name
                            break
                
                if target_location_instance is None:
                    for location in initial_states.values():
                        location_base = get_base_object_name(location)
                        if location_base == to_location_base:
                            target_location_instance = location
                            break
                
                # Check if target location is occupied
                object_at_target = None
                if target_location_instance:
                    for obj_name, location in initial_states.items():
                        if location == target_location_instance:
                            object_at_target = obj_name
                            break
                
                # Apply move (with swap if needed)
                if object_at_target and source_object_current_location:
                    # Swap positions
                    object_at_target_base = get_base_object_name(object_at_target)
                    current_bddl = temp_perturbator._replace_init_move(
                        current_bddl,
                        object_at_target_base,
                        get_base_object_name(source_object_current_location),
                        object_instance=object_at_target,
                        objects_config=objects_config
                    )
                    current_bddl = temp_perturbator._replace_init_move(
                        current_bddl,
                        from_object_base,
                        to_location_base,
                        object_instance=outcome['object_instance'],
                        objects_config=objects_config
                    )
                else:
                    # Target location is empty, just move
                    current_bddl = temp_perturbator._replace_init_move(
                        current_bddl,
                        from_object_base,
                        to_location_base,
                        object_instance=outcome['object_instance'],
                        objects_config=objects_config
                    )
                
                # Apply additional modifications if any
                if outcome.get('additional_modifications'):
                    current_bddl = temp_perturbator._apply_additional_modifications(
                        current_bddl,
                        outcome['additional_modifications']
                    )
                
                # Update task name
                task_name_parts.append(f"{outcome['object_instance']}_to_{to_location_base}")
            
            final_task_name = "_".join(task_name_parts)
            return final_task_name, current_bddl
        
        # Helper function to process cross combine mode
        def process_cross_combine_environment(operators_list, base_bddl_content, base_task_name, split_name, bddl_files_base, objects_config, id_to_config, expanded_base_tasks):
            """Process cross combine mode: cartesian product of all operator outcomes with conflict filtering."""
            # Resolve all operators (handle replicate_id)
            resolved_operators = []
            for op_config in operators_list:
                if isinstance(op_config, dict) and 'replicate_id' in op_config:
                    replicate_id = op_config['replicate_id']
                    if replicate_id not in id_to_config:
                        print(f"Warning: replicate_id '{replicate_id}' not found in stored configurations")
                        continue
                    resolved_operators.append(id_to_config[replicate_id].copy())
                else:
                    resolved_operators.append(op_config)
            
            if not resolved_operators:
                return
            
            # Generate all outcomes for each operator (from base state)
            operator_outcomes = []
            for op_config in resolved_operators:
                if op_config.get('type') == 'move':
                    if 'include_noop' not in op_config:
                        print(f"Warning: Missing required 'include_noop' for combine operator: {op_config}")
                        return
                    include_noop = op_config.get('include_noop', False)
                    outcomes = process_move_operation(
                        op_config, base_bddl_content, base_task_name,
                        split_name, bddl_files_base, objects_config
                    )
                    # Extract move_info from outcomes (new format includes move_info as 5th element)
                    op_outcomes = []
                    for outcome in outcomes:
                        if len(outcome) >= 5:
                            # New format: (task_name, bddl, is_swapped, swap_info, move_info)
                            move_info = outcome[4].copy()  # Make a copy to avoid modifying original
                            op_outcomes.append(move_info)
                        else:
                            # Fallback for old format - reconstruct move_info
                            swap_info = outcome[2] if len(outcome) > 2 else {}
                            op_outcomes.append({
                                'object_instance': swap_info.get('object_instance', 'unknown'),
                                'from_location': None,
                                'to_location': swap_info.get('to_location', 'unknown'),
                                'to_location_base': swap_info.get('to_location', 'unknown'),
                                'swapped_with': None,
                                'additional_modifications': {}
                            })
                    # Add a "no-op" option only when the operator allows it
                    if include_noop:
                        no_op_outcome = {
                            'object_instance': None,  # None indicates no-op
                            'from_location': None,
                            'to_location': None,
                            'to_location_base': None,
                            'swapped_with': None,
                            'additional_modifications': {},
                            'is_noop': True
                        }
                        op_outcomes.append(no_op_outcome)
                    
                    if op_outcomes:
                        operator_outcomes.append(op_outcomes)
            
            if not operator_outcomes:
                return
            
            # Generate cartesian product of all operator outcomes
            from itertools import product
            all_combinations = list(product(*operator_outcomes))
            
            # Filter out invalid combinations and apply valid ones
            for combination in all_combinations:
                # Filter out no-op outcomes for conflict detection
                active_outcomes = [outcome for outcome in combination if not outcome.get('is_noop', False)]
                
                # Skip if all operators are no-op (would just be base state)
                if not active_outcomes:
                    continue
                
                # Check for conflicts only among active outcomes
                has_conflict, conflict_reason = detect_move_conflicts(active_outcomes)
                if has_conflict:
                    continue  # Skip invalid combination
                
                # Apply only the active moves (no-op outcomes are skipped)
                final_task_name, final_bddl = apply_move_combination(
                    active_outcomes, base_bddl_content, base_task_name,
                    split_name, bddl_files_base, objects_config
                )
                
                if final_bddl != base_bddl_content:
                    swap_info = {
                        'base_task': base_task_name,
                        'cross_combine': True,
                        'operators': [op.get('type', 'unknown') for op in resolved_operators]
                    }
                    expanded_base_tasks.append((final_task_name, final_bddl, True, swap_info))
        
        # Helper function to process multi-step environment (list of operations)
        def process_multi_step_environment(step_list, base_bddl_content, base_task_name, split_name, bddl_files_base, objects_config, id_to_config, expanded_base_tasks):
            """Process a multi-step environment where each step builds on the previous."""
            # Start with the base BDDL content and task name
            current_states = [(base_task_name, base_bddl_content)]
            
            # Process each step sequentially
            for step_config in step_list:
                # Handle replicate_id: look up the stored configuration
                if isinstance(step_config, dict) and 'replicate_id' in step_config:
                    replicate_id = step_config['replicate_id']
                    if replicate_id not in id_to_config:
                        print(f"Warning: replicate_id '{replicate_id}' not found in stored configurations")
                        continue
                    # Use the stored configuration
                    step_config = id_to_config[replicate_id].copy()
                
                # If this step config has an id, store it for later reference
                if isinstance(step_config, dict) and 'id' in step_config:
                    config_id = step_config['id']
                    # Store a copy without the id field for reuse
                    config_copy = {k: v for k, v in step_config.items() if k != 'id'}
                    id_to_config[config_id] = config_copy
                
                # Process the step for each current state
                next_states = []
                for current_task_name, current_bddl_content in current_states:
                    # Process this move operation on the current BDDL content
                    if isinstance(step_config, dict) and step_config.get('type') == 'move':
                        include_noop = step_config.get('include_noop', False)
                        
                        step_tasks = process_move_operation(
                            step_config, current_bddl_content, current_task_name,
                            split_name, bddl_files_base, objects_config
                        )
                        
                        # Add all results from this step as inputs for the next step
                        # Handle both old format (4 elements) and new format (5 elements)
                        if step_tasks:
                            next_states.extend([(name, bddl) for name, bddl, _, _, *rest in step_tasks])
                        
                        # If include_noop is true, also add the no-op path (current state unchanged)
                        if include_noop:
                            next_states.append((current_task_name, current_bddl_content))
                        elif not step_tasks:
                            # If no results and no no-op, keep the current state (in case it's the last step)
                            next_states.append((current_task_name, current_bddl_content))
                    else:
                        # Unknown step type, keep current state
                        next_states.append((current_task_name, current_bddl_content))
                
                # Update current states for next iteration
                current_states = next_states
            
            # After processing all steps, add all final results
            for final_task_name, final_bddl_content in current_states:
                if final_bddl_content != base_bddl_content:
                    swap_info = {
                        'base_task': base_task_name,
                        'multi_step': True
                    }
                    expanded_base_tasks.append((final_task_name, final_bddl_content, True, swap_info))
        
        # Create expanded list of base tasks including swapped versions
        # Format: list of (base_task_name, bddl_content, is_swapped, swap_info) tuples
        # where swap_info is None for original tasks, or dict with swap details for swapped tasks
        expanded_base_tasks = []
        for base_task_name in valid_base_tasks:
            # Add original base task
            bddl_path = os.path.join(bddl_files_base, split_name, base_task_name + '.bddl')
            with open(bddl_path, "r", encoding="utf-8") as f:
                original_bddl_content = f.read()
            expanded_base_tasks.append((base_task_name, original_bddl_content, False, None))
            
            # Create swapped versions for each environment configuration
            for env_config in new_environments:
                # New format: env_config is a dict with 'combine' and 'operators'
                if isinstance(env_config, dict):
                    operators_list = env_config.get('operators', [])
                    combine_mode = env_config.get('combine', None)
                    
                    # Handle single operator case (no combine needed)
                    if len(operators_list) == 1:
                        # Resolve operator (handle replicate_id)
                        op_config = operators_list[0]
                        if isinstance(op_config, dict) and 'replicate_id' in op_config:
                            replicate_id = op_config['replicate_id']
                            if replicate_id not in id_to_config:
                                print(f"Warning: replicate_id '{replicate_id}' not found in stored configurations")
                                continue
                            op_config = id_to_config[replicate_id].copy()
                        
                        # Process single operator
                        if op_config.get('type') == 'move':
                            new_tasks = process_move_operation(
                                op_config, original_bddl_content, base_task_name,
                                split_name, bddl_files_base, split_objects_config
                            )
                            # Extract just the first 4 elements for compatibility
                            for task in new_tasks:
                                task_name, bddl, is_swapped, swap_info = task[:4]
                                expanded_base_tasks.append((task_name, bddl, is_swapped, swap_info))
                    
                    # Handle multiple operators
                    elif len(operators_list) > 1:
                        if combine_mode in ('stack', 'cross'):
                            missing_noop = False
                            for op_config in operators_list:
                                resolved_config = op_config
                                if isinstance(op_config, dict) and 'replicate_id' in op_config:
                                    replicate_id = op_config['replicate_id']
                                    if replicate_id not in id_to_config:
                                        print(f"Warning: replicate_id '{replicate_id}' not found in stored configurations")
                                        missing_noop = True
                                        break
                                    resolved_config = id_to_config[replicate_id]
                                if not isinstance(resolved_config, dict) or 'include_noop' not in resolved_config:
                                    print(f"Warning: Missing required 'include_noop' for combine operator: {resolved_config}")
                                    missing_noop = True
                                    break
                            if missing_noop:
                                continue
                        if combine_mode == 'stack':
                            # Sequential application (reuse existing function)
                            process_multi_step_environment(
                                operators_list, original_bddl_content, base_task_name, split_name,
                                bddl_files_base, split_objects_config, id_to_config, expanded_base_tasks
                            )
                        elif combine_mode == 'cross':
                            # Cartesian product with conflict filtering
                            process_cross_combine_environment(
                                operators_list, original_bddl_content, base_task_name, split_name,
                                bddl_files_base, split_objects_config, id_to_config, expanded_base_tasks
                            )
                        else:
                            print(f"Warning: Unknown combine mode '{combine_mode}' or missing combine for multiple operators")
                    
                    # Legacy format support: list of operations (backward compatibility)
                elif isinstance(env_config, list):
                    # Process each step sequentially, building on previous results
                    process_multi_step_environment(
                        env_config, original_bddl_content, base_task_name, split_name,
                        bddl_files_base, split_objects_config, id_to_config, expanded_base_tasks
                    )
        
        # Get naming convention for this split (if specified)
        naming_convention = split_config.get('naming_convention', None)
        
        # Collect objects from ALL existing BDDL files (not just base tasks)
        # This ensures we have all objects needed to generate all existing tasks.
        # For libero_goal, different tasks use different objects (e.g., wine_bottle_1 for some tasks,
        # akita_black_bowl_1 for others), so we need objects from all existing tasks.
        # For libero_spatial, we still use all objects from all existing tasks, but we iterate over
        # base tasks to get different initial locations.
        all_objects_from_existing_bddls = objects_in_existing_bddls.copy() if objects_in_existing_bddls else set()
        
        # Filter to only include objects that match (by base name) objects in split_objects_config
        # This ensures we only generate tasks for objects that are configured in the YAML
        # Sort to ensure deterministic ordering
        all_available_objects = sorted([
            obj_name for obj_name in all_objects_from_existing_bddls
            if match_object_to_config(obj_name, split_objects_config) is not None
        ])
        
        # Get chaining configuration from split config (with fallback to default)
        chaining_config = split_config.get('chaining_config', {'max_chain_length': 1, 'reasonable_only': False})
        chaining_source_dirs = chaining_config.get('source_dirs', None)
        chaining_existing_only = chaining_source_dirs is not None and 'existing' in chaining_source_dirs
        generate_second_step_tasks = bool(chaining_config.get('generate_second_step_tasks', False))
        
        # Generate all valid task combinations (including chained tasks) once for all base tasks
        # all_available_objects now contains instance names from ALL existing BDDL files
        chained_task_specs = generate_chained_task_combinations(
            all_available_objects, split_objects_config, operators_config, chaining_config
        )
        
        # Sort chained_task_specs to ensure deterministic processing order
        # Sort by a canonical representation: convert actions to a sortable string
        def get_task_sort_key(chained_task):
            """Generate a sortable key for a chained task to ensure deterministic ordering."""
            actions = chained_task['actions']
            # Create a canonical string representation of the task
            parts = []
            for action in actions:
                action_type = action['type']
                if action_type == 'On':
                    variant = action.get('variant', 'grasp')
                    obj1_base = get_base_object_name(action['obj1'])
                    obj2_base = get_base_object_name(action['obj2'])
                    parts.append(f"{action_type}:{variant}:{obj1_base}:{obj2_base}")
                elif action_type in ('In', 'PlaceIn'):
                    obj1_base = get_base_object_name(action['obj1'])
                    obj2_base = get_base_object_name(action['obj2'])
                    parts.append(f"{action_type}:{obj1_base}:{obj2_base}")
                else:
                    obj1_base = get_base_object_name(action['obj1'])
                    parts.append(f"{action_type}:{obj1_base}")
            return "|".join(parts)
        
        # Sort so non-chained (single-step) tasks are generated first; chained tasks get suffixes on name collision
        chained_task_specs = sorted(
            chained_task_specs,
            key=lambda t: (t.get('is_chained', len(t['actions']) > 1), get_task_sort_key(t))
        )

        def _apply_first_action_to_init(
            base_bddl: str,
            first_action: Dict,
            split_objects_cfg: Dict
        ) -> str:
            """
            Construct a mid-state BDDL by applying only the first action of a 2-step chain
            to the base BDDL :init section.
            """
            task_type = first_action['type']

            # Use a temporary perturbator that operates directly on provided content
            temp_perturbator = BDDLPerturbator(
                split_name,
                base_task_name,
                bddl_files_base=bddl_files_base,
                bddl_content=base_bddl
            )

            # For On/In actions (including push variant on On), move the relevant object
            if task_type == 'On':
                obj1_name = first_action['obj1']
                obj2_name = first_action['obj2']
                from_object_base = get_base_object_name(obj1_name)
                to_location_base = get_base_object_name(obj2_name)
                return temp_perturbator._replace_init_move(
                    base_bddl,
                    from_object_base,
                    to_location_base,
                    object_instance=obj1_name,
                    objects_config=split_objects_cfg
                )
            if task_type == 'In':
                # In uses obj2 as the moved object and obj1 as the container
                container = first_action['obj1']
                obj_to_place = first_action['obj2']
                from_object_base = get_base_object_name(obj_to_place)
                to_location_base = get_base_object_name(container)
                return temp_perturbator._replace_init_move(
                    base_bddl,
                    from_object_base,
                    to_location_base,
                    object_instance=obj_to_place,
                    objects_config=split_objects_cfg
                )

            # For Open/Turnon, encode the effect directly in :init via additional predicates
            if task_type == 'Open':
                obj = first_action['obj1']
                mods = {'init_states': [['Open', obj]]}
                return temp_perturbator._apply_additional_modifications(base_bddl, mods)
            if task_type == 'Turnon':
                obj = first_action['obj1']
                mods = {'init_states': [['Turnon', obj]]}
                return temp_perturbator._apply_additional_modifications(base_bddl, mods)

            # Unknown first-action type for mid-state generation – leave unchanged
            return base_bddl
        
        for base_task_name, bddl_content, is_swapped, swap_info in expanded_base_tasks:
            # Parse the base BDDL content (may be original or swapped)
            base_parser = BDDLParser(bddl_content)
            
            # Use original base_task_name for BDDLPerturbator (it needs the original filename to load)
            # But we'll track if this is a swapped version
            base_task_filename = base_task_name if not is_swapped else swap_info['base_task']
            
            for chained_task in chained_task_specs:
                # Get the list of actions in this task
                actions = chained_task['actions']
                is_chained = chained_task.get('is_chained', len(actions) > 1)

                # PlaceIn actions require the container to already be open, so they are only
                # meaningful as derived second-step tasks (with the drawer pre-opened in init).
                # Skip generating standalone BDDL files for single-action PlaceIn tasks.
                if len(actions) == 1 and actions[0]['type'] == 'PlaceIn':
                    continue

                # If chaining is configured to use only existing environments, skip swapped bases
                # only for chained tasks (2+ steps). Single-step tasks still use all bases so that
                # variants like put_the_bowl_on_the_plate_2/_3 (different inits) are generated.
                if chaining_existing_only and is_swapped and is_chained:
                    continue
                
                # Build up current object locations by simulating actions
                # Start with initial state from base BDDL (object -> location mapping)
                current_object_locations = {}
                if base_parser.initial_states:
                    current_object_locations = base_parser.initial_states.copy()
                
                # Validate each action against the current state (after previous actions)
                should_skip_task = False
                for action in actions:
                    if action['type'] == 'On' and action.get('variant', 'grasp') == 'grasp':
                        obj1_name = action['obj1']  # object being moved
                        obj2_name = action['obj2']  # placement target
                        
                        # Get current location of obj1 (may have been moved by previous actions)
                        obj1_current_location = current_object_locations.get(obj1_name)
                        
                        # Skip if placement target is the same as current location
                        if obj1_current_location == obj2_name:
                            should_skip_task = True
                            break
                        
                        # Check if target location is already occupied by another object
                        for other_obj, other_obj_location in current_object_locations.items():
                            if other_obj != obj1_name and other_obj_location == obj2_name:
                                # Target location is already occupied by another object
                                should_skip_task = True
                                break
                        if should_skip_task:
                            break
                        
                        # Update state: obj1 is now at obj2 (simulate the action)
                        current_object_locations[obj1_name] = obj2_name
                        
                    elif action['type'] == 'In':
                        obj2_name = action['obj2']  # object to place
                        obj1_name = action['obj1']  # container
                        
                        # Get current location of obj2 (may have been moved by previous actions)
                        obj2_current_location = current_object_locations.get(obj2_name)
                        
                        # Skip if placement target (container) is the same as current location
                        if obj2_current_location == obj1_name:
                            should_skip_task = True
                            break
                        
                        # Check if target container is already occupied by another object
                        for other_obj, other_obj_location in current_object_locations.items():
                            if other_obj != obj2_name and other_obj_location == obj1_name:
                                # Target container is already occupied by another object
                                should_skip_task = True
                                break
                        if should_skip_task:
                            break
                        
                        # Update state: obj2 is now in obj1 (simulate the action)
                        current_object_locations[obj2_name] = obj1_name
                
                if should_skip_task:
                    continue  # Skip this task - invalid placement detected
                
                # Get initial location of the first action's object from the current base BDDL
                # For libero_spatial, each base task has different initial locations for objects,
                # so we must use the initial location from the current base task we're iterating over.
                # This ensures we generate tasks that preserve the initial state from that base.
                initial_location = None
                initial_location_relation = None  # 'On' or 'In'
                target_initial_location = None
                target_initial_location_relation = None  # 'On' or 'In'
                if actions:
                    first_action = actions[0]
                    if first_action['type'] == 'On' and first_action.get('variant', 'grasp') == 'grasp':
                        obj1_name = first_action['obj1']
                        obj2_name = first_action['obj2']  # placement target
                        
                        # Use the initial location from the current base BDDL
                        result = base_parser.get_initial_location_relation(obj1_name)
                        
                        if result is not None:
                            initial_location, initial_location_relation = result
                        
                        # Get target object's initial location for naming (if configured)
                        target_result = base_parser.get_initial_location_relation(obj2_name)
                        if target_result is not None:
                            target_initial_location, target_initial_location_relation = target_result
                    elif first_action['type'] == 'In':
                        obj2_name = first_action['obj2']  # object to place
                        obj1_name = first_action['obj1']  # container
                        
                        # Use the initial location from the current base BDDL
                        result = base_parser.get_initial_location_relation(obj2_name)
                        
                        if result is not None:
                            initial_location, initial_location_relation = result
                        
                        # Get target object's (container's) initial location for naming (if configured)
                        target_result = base_parser.get_initial_location_relation(obj1_name)
                        if target_result is not None:
                            target_initial_location, target_initial_location_relation = target_result
                
                # Only use initial_location for naming convention if naming convention requires it
                initial_location_for_naming = None
                initial_location_relation_for_naming = None
                if naming_convention and naming_convention.get('include_initial_location', False):
                    initial_location_for_naming = initial_location
                    initial_location_relation_for_naming = initial_location_relation
                
                # Only use target_initial_location for naming convention if configured
                target_initial_location_for_naming = None
                target_initial_location_relation_for_naming = None
                if naming_convention and naming_convention.get('include_target_initial_location_for_objects'):
                    target_initial_location_for_naming = target_initial_location
                    target_initial_location_relation_for_naming = target_initial_location_relation
                
                # Generate filename and language (using chained helper for multi-action tasks)
                cur_filename_lang, cur_bddl_lang = generate_chained_filename_and_lang(
                    chained_task, split_objects_config, operators_config, naming_convention, initial_location_for_naming, initial_location_relation_for_naming, target_initial_location_for_naming, target_initial_location_relation_for_naming
                )
                
                # If --only-tasks is specified, skip tasks not in the list
                if only_tasks_set is not None and cur_filename_lang not in only_tasks_set:
                    continue
                
                is_existing_task = cur_filename_lang in existing_tasks_filenames
                
                # Generate execution steps (concatenated for chained tasks)
                execution_steps = generate_chained_execution_steps(
                    chained_task, split_objects_config, operators_config
                )
                
                # Record additional information about the task
                cur_metadata = {
                    'execution_steps': execution_steps
                }
                
                # Generate BDDL goal and objects of interest (using chained helper)
                bddl_operator, bddl_objects_of_interest = generate_chained_bddl_goal(
                    chained_task, split_objects_config, operators_config
                )
                
                # Print task info
                task_types = [a['type'] for a in actions]
                print(f"{'->'.join(task_types)}", cur_filename_lang, cur_bddl_lang, bddl_operator)
                print(execution_steps)
                print()
                
                # Generate BDDL content for the full chained task
                # Use swapped content if this is a swapped base task, otherwise None (will load from file)
                bddl_perturbator = BDDLPerturbator(
                    split_name, 
                    base_task_filename, 
                    # Enforce consistent sentence-case language: only first char uppercase, rest lowercase.
                    # This prevents mixed-case like "... and Put ..." coming from operator prefixes in YAML.
                    new_lang=to_sentence_case(cur_bddl_lang),
                    new_goal=bddl_operator, 
                    new_objs_of_interest=bddl_objects_of_interest,
                    bddl_files_base=bddl_files_base,
                    bddl_content=bddl_content if is_swapped else None
                )
                new_bddl_content = bddl_perturbator.perturb()

                # Optionally generate second-step-only single-action task built on the mid-state
                if (
                    generate_second_step_tasks
                    and not is_swapped
                    and is_chained
                    and len(actions) == 2
                ):
                    first_action = actions[0]
                    second_action = actions[1]
                    mid_bddl_content = _apply_first_action_to_init(bddl_content, first_action, split_objects_config)

                    # Determine the object(s) moved/acted on by the first action
                    first_action_objects = set()
                    if first_action.get('obj1'):
                        first_action_objects.add(get_base_object_name(first_action['obj1']))
                    if first_action.get('obj2'):
                        first_action_objects.add(get_base_object_name(first_action['obj2']))

                    # Determine the object(s) grasped/moved by the second action.
                    # For PlaceIn, obj1 is the container (not grasped) and obj2 is the grasped object.
                    second_action_grasped_objects = set()
                    if second_action['type'] == 'PlaceIn':
                        if second_action.get('obj2'):
                            second_action_grasped_objects.add(get_base_object_name(second_action['obj2']))
                    elif second_action.get('obj1'):
                        second_action_grasped_objects.add(get_base_object_name(second_action['obj1']))

                    # Skip second-step tasks where the second action re-uses an object from the first action.
                    # This avoids chains like "place wine on plate → move wine (now on plate) to cabinet".
                    second_step_reuses_first_object = bool(first_action_objects & second_action_grasped_objects)

                    # Only proceed if the first action actually changed the init state
                    if mid_bddl_content != bddl_content and not second_step_reuses_first_object:
                        second_step_chained = {'actions': [second_action]}

                        # Naming and language for the second-step-only task
                        second_filename_lang, second_bddl_lang = generate_chained_filename_and_lang(
                            second_step_chained,
                            split_objects_config,
                            operators_config,
                            naming_convention,
                            None,
                            None,
                            None,
                            None
                        )

                        # Respect --only-tasks filter
                        if only_tasks_set is None or second_filename_lang in only_tasks_set:
                            second_execution_steps = generate_chained_execution_steps(
                                second_step_chained,
                                split_objects_config,
                                operators_config
                            )
                            second_bddl_goal, second_objs_of_interest = generate_chained_bddl_goal(
                                second_step_chained,
                                split_objects_config,
                                operators_config
                            )
                            
                            # Build metadata with derivation info
                            # Note: we also record the originating chained task name so that
                            # downstream view builders (e.g., chain_secondstep_view) can
                            # inherit additional metadata such as grasps_from.
                            second_metadata = {
                                'execution_steps': second_execution_steps,
                                'derived_from_chain': True,
                                'derived_chain_length': 2,
                                'derived_chain_step': 2,
                                'derived_base_task': base_task_name,
                                'derived_full_chain_types': [first_action['type'], second_action['type']],
                                'derived_from_full_chain_task': cur_filename_lang,
                            }

                            # Apply skip_generation_for rules to the second-step-only task as well.
                            # For init-based conditions, we should use the mid-state init (after the first action).
                            # Only skip the second-step task; do not skip the full chained task (no continue).
                            skip_second_step = False
                            if skip_generation_for:
                                second_parser = BDDLParser(mid_bddl_content)
                                if should_skip_task_generation(
                                    second_step_chained,
                                    skip_generation_for,
                                    second_metadata,
                                    second_execution_steps,
                                    existing_tasks_metadata,
                                    initial_states=second_parser.initial_states,
                                    open_predicates=_parse_open_turnon_from_bddl(mid_bddl_content),
                                ):
                                    print(
                                        f"Note: Second-step-only task '{second_filename_lang}' "
                                        f"matches skip_generation_for conditions. Skipping second-step-only generation."
                                    )
                                    skip_second_step = True
                            
                            if not skip_second_step:
                                # Use the same base filename as the second action and rely on suffixing for collisions
                                base_second_name = second_filename_lang
                                second_final_filename = base_second_name
                                
                                # Ensure uniqueness across existing and newly generated tasks by suffixing
                                suffix_num = 2
                                while (
                                    second_final_filename in existing_tasks_filenames_all
                                    or second_final_filename in generated_task_names
                                ):
                                    second_final_filename = f"{base_second_name}_{suffix_num}"
                                    suffix_num += 1
                                
                                # Save metadata as extra task
                                second_metadata['is_existing_task'] = False
                                task_metadata_extra[second_final_filename] = second_metadata
                                generated_task_names.add(second_final_filename)
                                
                                # Build final BDDL for the second-step-only task, using mid-state as content
                                second_perturbator = BDDLPerturbator(
                                    split_name,
                                    base_task_filename,
                                    new_lang=to_sentence_case(second_bddl_lang),
                                    new_goal=second_bddl_goal,
                                    new_objs_of_interest=second_objs_of_interest,
                                    bddl_files_base=bddl_files_base,
                                    bddl_content=mid_bddl_content
                                )
                                second_bddl_content = second_perturbator.perturb()
                                
                                # Save BDDL into extra directory
                                out_second_bddl_dir = out_bddl_dir_extra
                                out_second_bddl_filename = os.path.join(
                                    out_second_bddl_dir,
                                    second_final_filename + '.bddl'
                                )
                                Path(out_second_bddl_filename).parent.mkdir(parents=True, exist_ok=True)
                                with Path(out_second_bddl_filename).open("w", encoding="utf-8") as f:
                                    f.write(second_bddl_content)
                                
                                # Track for functional equivalence checks
                                generated_task_info[second_final_filename] = {
                                    'bddl_content': second_bddl_content,
                                    'execution_steps': second_execution_steps,
                                    'is_chained': False
                                }
                
                # An Open→PlaceIn chain on the same container (e.g., open middle drawer then put
                # bowl in middle drawer) is identical in robot behavior to the existing In task
                # (open middle drawer and put bowl inside). The only difference is naming convention.
                # Skip writing the full chain BDDL so we don't create a duplicate task; the
                # second-step task (drawer pre-opened) was already generated above.
                if (
                    len(actions) == 2
                    and actions[0]['type'] == 'Open'
                    and actions[1]['type'] == 'PlaceIn'
                    and actions[0]['obj1'] == actions[1]['obj1']
                ):
                    continue

                # Handle name conflicts by adding suffixes when init sections don't match
                final_filename = cur_filename_lang
                if is_existing_task:
                    existing_tasks_filename_matches.add(cur_filename_lang)
                    # Check if existing BDDL file exists and compare init sections
                    existing_bddl_path = os.path.join(out_bddl_dir_original, cur_filename_lang + '.bddl')
                    if os.path.exists(existing_bddl_path):
                        with open(existing_bddl_path, 'r', encoding='utf-8') as f:
                            existing_content = f.read()
                        # Check functional equivalence and instance-level match
                        functional_equivalent, instance_level_match = are_bddl_functionally_equivalent(existing_content, new_bddl_content)
                        if functional_equivalent:
                            if instance_level_match:
                                # Functionally equivalent and exact match on instance level
                                # Sanity check: verify full BDDL content matches exactly
                                if existing_content != new_bddl_content:
                                    print(f"\nERROR: Generated BDDL content does not match existing file for {existing_bddl_path}")
                                    print("\n" + "="*80)
                                    print("DIFF (existing file vs generated content):")
                                    print("="*80)
                                    existing_lines = existing_content.splitlines(keepends=True)
                                    new_lines = new_bddl_content.splitlines(keepends=True)
                                    diff = difflib.unified_diff(
                                        existing_lines,
                                        new_lines,
                                        fromfile=f'existing: {existing_bddl_path}',
                                        tofile='generated',
                                        lineterm=''
                                    )
                                    for line in diff:
                                        print(line, end='')
                                    print("="*80 + "\n")
                                    assert False, f"Generated BDDL content does not match existing file for {existing_bddl_path}"
                                # Content matches exactly, add to existing tasks metadata before skipping
                                cur_metadata['is_existing_task'] = True
                                task_metadata_existing[cur_filename_lang] = cur_metadata
                                continue
                            else:
                                # Functionally equivalent but different on instance level (e.g., bowl_1 vs bowl_2)
                                # Skip generating this task without writing to task_metadata_existing
                                print(f"Note: Task '{cur_filename_lang}' is functionally equivalent to existing task but differs on instance level. Skipping generation.")
                                continue
                        else:
                            # Init sections don't match
                            should_skip_duplicate = False
                            
                            if duplicate_mode == 'false':
                                # Skip generating duplicate task
                                should_skip_duplicate = True
                                print(f"Note: Task '{cur_filename_lang}' exists but has different init configuration. Skipping (duplicate_mode=false).")
                            elif duplicate_mode == 'only_relevant_objects':
                                # Check if only non-interacted objects differ
                                if only_relevant_entities_and_locations_differ(existing_content, new_bddl_content, execution_steps):
                                    # Only non-interacted objects differ, skip duplicate
                                    should_skip_duplicate = True
                                    print(f"Note: Task '{cur_filename_lang}' exists but has different init configuration. Only non-interacted objects differ. Skipping (duplicate_mode=only_relevant_objects).")
                                else:
                                    # Interacted objects differ, allow duplicate
                                    print(f"Note: Task '{cur_filename_lang}' exists but has different init configuration for interacted objects. Allowing duplicate (duplicate_mode=only_relevant_objects).")
                            
                            if should_skip_duplicate:
                                continue
                            
                            # Add suffix to create a new task (duplicate_mode is 'true' or 'only_relevant_objects' with interacted objects differing)
                            suffix_num = 2
                            while f"{cur_filename_lang}_{suffix_num}" in existing_tasks_filenames_all or f"{cur_filename_lang}_{suffix_num}" in generated_task_names:
                                suffix_num += 1
                            final_filename = f"{cur_filename_lang}_{suffix_num}"
                            print(f"Note: Task '{cur_filename_lang}' exists but has different init configuration. Creating as '{final_filename}'")
                            # This is now a new task, not an existing one
                            is_existing_task = False
                    else:
                        print(f"WARNING: Existing BDDL file not found: {existing_bddl_path}")
                        # If file doesn't exist but name is in existing_tasks_filenames, treat as new task
                        is_existing_task = False
                
                # Check for functional equivalence with already-generated tasks in the same run
                if not is_existing_task:
                    skip_task = False
                    for existing_generated_filename, existing_task_info in generated_task_info.items():
                        existing_bddl_content = existing_task_info['bddl_content']
                        existing_execution_steps = existing_task_info['execution_steps']
                        existing_is_chained = existing_task_info['is_chained']
                        
                        functional_equivalent, instance_level_match = are_bddl_functionally_equivalent(
                            existing_bddl_content, new_bddl_content
                        )
                        if functional_equivalent:
                            # For chained tasks, also check if execution order matches
                            # Tasks with different execution orders should be considered different
                            if is_chained and existing_is_chained:
                                # Both are chained tasks - compare execution steps
                                if execution_steps != existing_execution_steps:
                                    # Different execution order - these are different tasks
                                    continue
                                # Same execution order and functionally equivalent - skip
                            elif not is_chained and not existing_is_chained:
                                # Both are single-action tasks - functionally equivalent means same task
                                pass
                            else:
                                # One is chained, one is not - different tasks
                                continue
                            
                            # Skip this task - it's functionally equivalent to one already generated
                            print(f"Note: Task '{cur_filename_lang}' is functionally equivalent to already-generated task '{existing_generated_filename}'. Skipping generation.")
                            skip_task = True
                            break
                    if skip_task:
                        continue
                
                # Check for conflicts with newly generated tasks in the same run
                if final_filename in generated_task_names:
                    should_skip_duplicate = False
                    
                    if duplicate_mode == 'false':
                        # Skip generating duplicate task
                        should_skip_duplicate = True
                        print(f"Note: Task name conflict detected for '{final_filename}'. Skipping (duplicate_mode=false).")
                    elif duplicate_mode == 'only_relevant_objects':
                        # Get the existing task's BDDL content and execution steps
                        existing_task_info = generated_task_info.get(final_filename)
                        if existing_task_info:
                            existing_bddl_content = existing_task_info['bddl_content']
                            # Check if only non-interacted objects differ
                            if only_relevant_entities_and_locations_differ(existing_bddl_content, new_bddl_content, execution_steps):
                                # Only non-interacted objects differ, skip duplicate
                                should_skip_duplicate = True
                                print(f"Note: Task name conflict detected for '{final_filename}'. Only non-interacted objects differ. Skipping (duplicate_mode=only_relevant_objects).")
                            else:
                                # Interacted objects differ, allow duplicate
                                print(f"Note: Task name conflict detected for '{final_filename}'. Interacted objects differ. Allowing duplicate (duplicate_mode=only_relevant_objects).")
                        else:
                            # Can't check, default to allowing duplicate
                            print(f"Note: Task name conflict detected for '{final_filename}'. Cannot check relevant objects, allowing duplicate.")
                    
                    if should_skip_duplicate:
                        continue
                    
                    # Find an available suffix (duplicate_mode is 'true' or 'only_relevant_objects' with interacted objects differing)
                    suffix_num = 2
                    base_name = final_filename
                    # If it already has a suffix, extract the base name
                    if '_' in base_name and base_name.split('_')[-1].isdigit():
                        parts = base_name.rsplit('_', 1)
                        base_name = parts[0]
                        suffix_num = int(parts[1]) + 1
                    while f"{base_name}_{suffix_num}" in existing_tasks_filenames_all or f"{base_name}_{suffix_num}" in generated_task_names:
                        suffix_num += 1
                    final_filename = f"{base_name}_{suffix_num}"
                    print(f"Note: Task name conflict detected. Using '{final_filename}' instead")
                
                # Initialize pattern override variables (used in both actions_from and grasps_from sections)
                pattern_override_matched = False
                pattern_grasps_from = None
                pattern_overrides_dict = None
                
                # Auto-infer actions_from (can be overridden via task_pattern_overrides or task_overrides)
                # This must be done AFTER name conflict handling so we use final_filename for task overrides
                # For chained tasks, only the first action determines actions_from
                if not is_existing_task:
                    # Check for pattern-based override first
                    for pattern_override in task_pattern_overrides:
                        pattern = pattern_override.get('pattern', {})
                        overrides = pattern_override.get('overrides', {})
                        
                        if task_matches_pattern(chained_task, pattern, base_parser.initial_states):
                            pattern_override_matched = True
                            pattern_overrides_dict = overrides
                            pattern_grasps_from = overrides.get('grasps_from')
                            break
                    
                    # If pattern matched, use pattern override
                    if pattern_override_matched:
                        # Check if actions_from key exists in overrides (even if value is None/null)
                        if 'actions_from' in pattern_overrides_dict:
                            actions_from_override = pattern_overrides_dict['actions_from']
                            # actions_from_override can be null/None to force no actions_from
                            if actions_from_override not in [None, 'null']:
                                # Validate that override task exists in existing tasks
                                if actions_from_override not in existing_tasks_metadata:
                                    raise ValueError(f"task_pattern_overrides actions_from references task '{actions_from_override}' which is not in existing tasks")
                                actions_from = actions_from_override
                            else:
                                # Explicitly set to None (no actions_from)
                                actions_from = None
                        else:
                            # No override specified, use default inference
                            new_task_initial_states = base_parser.initial_states
                            actions_from = find_actions_from(
                                existing_tasks_metadata, 
                                execution_steps,
                                default_actions_from_strategy=default_actions_from_strategy,
                                initial_states=new_task_initial_states
                            )
                    else:
                        # Check for task-specific override by name (use final_filename, not cur_filename_lang)
                        actions_from_override = task_overrides.get(final_filename, {}).get('actions_from')
                        if actions_from_override:
                            # Validate that override task exists in existing tasks
                            if actions_from_override not in existing_tasks_metadata:
                                raise ValueError(f"task_overrides['{final_filename}']['actions_from'] references task '{actions_from_override}' which is not in existing tasks")
                            actions_from = actions_from_override
                        else:
                            # Get full init state from base BDDL for matching
                            new_task_initial_states = base_parser.initial_states
                            actions_from = find_actions_from(
                                existing_tasks_metadata, 
                                execution_steps,
                                default_actions_from_strategy=default_actions_from_strategy,
                                initial_states=new_task_initial_states
                            )
                    
                    if actions_from is not None:
                        cur_metadata['actions_from'] = actions_from
                
                # Auto-infer grasps_from for all grasped objects in the task
                # But skip if the grasp is already covered by actions_from_steps
                # actions_from_steps corresponds semantically one-to-one with the first entries of execution_steps
                grasps_from = {}
                
                # Compute which steps from actions_from will be replayed (same logic as later in the code)
                # This determines how many steps are covered by action replay
                num_replayed_steps = 0
                if 'actions_from' in cur_metadata:
                    actions_from_task = cur_metadata['actions_from']
                    actions_from_task_steps = existing_tasks_metadata[actions_from_task].get('execution_steps', [])
                    
                    # Compare steps sequentially to find matching steps (by exact object instances)
                    # Since we match by exact instances and init states, steps must match exactly
                    min_len = min(len(execution_steps), len(actions_from_task_steps))
                    for i in range(min_len):
                        cur_step = execution_steps[i]
                        from_step = actions_from_task_steps[i]
                        
                        cur_stage, cur_obj = parse_execution_step(cur_step)
                        from_stage, from_obj = parse_execution_step(from_step)
                        
                        if cur_stage != from_stage:
                            break
                        
                        # For Place steps, check both objects by exact instance
                        if cur_stage == 'Place':
                            cur_parts = cur_step.split()
                            from_parts = from_step.split()
                            if len(cur_parts) >= 3 and len(from_parts) >= 3:
                                cur_obj1 = cur_parts[1]  # object being placed
                                cur_obj2 = cur_parts[2]  # target location
                                from_obj1 = from_parts[1]
                                from_obj2 = from_parts[2]
                                # Compare by exact instance
                                if cur_obj1 == from_obj1 and cur_obj2 == from_obj2:
                                    num_replayed_steps += 1
                                else:
                                    break
                            else:
                                break
                        elif cur_obj and from_obj:
                            # Compare by exact instance
                            if cur_obj == from_obj:
                                num_replayed_steps += 1
                            else:
                                break
                        else:
                            num_replayed_steps += 1
                
                # Collect all objects that need to be grasped in this task, along with their execution step indices
                # Each entry is (obj_to_grasp, execution_step_index, place_location)
                objects_to_grasp_with_indices = []
                step_index = 0
                for action in actions:
                    if action['type'] == 'On' and action.get('variant', 'grasp') == 'grasp':
                        # Grasp is at step_index, Place is at step_index + 1
                        objects_to_grasp_with_indices.append((action['obj1'], step_index, action.get('obj2')))
                        step_index += 2  # Grasp + Place
                    elif action['type'] == 'In':
                        # Open is at step_index, Grasp is at step_index + 1, Place is at step_index + 2
                        objects_to_grasp_with_indices.append((action['obj2'], step_index + 1, action.get('obj1')))
                        step_index += 3  # Open + Grasp + Place
                    elif action['type'] == 'Open':
                        step_index += 1
                    elif action['type'] == 'Turnon':
                        step_index += 1
                
                # Check each object to grasp
                for obj_to_grasp, execution_step_index, obj_place_location in objects_to_grasp_with_indices:
                    # Check if this grasp is already covered by actions_from_steps
                    # The grasp is covered if:
                    # 1. The execution step index is within the replayed steps
                    # 2. The corresponding step in actions_from is a Grasp of the same object (by exact instance)
                    grasp_covered_by_actions_from = False
                    if 'actions_from' in cur_metadata and execution_step_index < num_replayed_steps:
                        actions_from_task = cur_metadata['actions_from']
                        actions_from_task_steps = existing_tasks_metadata[actions_from_task].get('execution_steps', [])
                        if execution_step_index < len(actions_from_task_steps):
                            from_step = actions_from_task_steps[execution_step_index]
                            from_stage, from_obj = parse_execution_step(from_step)
                            if from_stage == 'Grasp' and from_obj:
                                # Match by exact instance (since we match by exact instances and init states)
                                if obj_to_grasp == from_obj:
                                    grasp_covered_by_actions_from = True
                    
                    if not grasp_covered_by_actions_from:
                        grasp_source = None
                        # Check for pattern-based override first
                        pattern_grasp_override = None
                        if pattern_override_matched and pattern_grasps_from is not None:
                            # Check if this object has a pattern-based grasp override
                            obj_base = get_base_object_name(obj_to_grasp)
                            if obj_base in pattern_grasps_from:
                                grasp_info = pattern_grasps_from[obj_base]
                                if isinstance(grasp_info, dict) and 'task' in grasp_info:
                                    pattern_grasp_override = grasp_info['task']
                        
                        # Check for task-specific override by name (use final_filename, not cur_filename_lang)
                        task_override = task_overrides.get(final_filename, {}).get('grasps_from')
                        
                        # Use pattern override if available, otherwise use name-based override
                        final_grasp_override = pattern_grasp_override if pattern_grasp_override else task_override
                        
                        # If pattern override matched and we have a grasp override, use it directly
                        if pattern_override_matched and pattern_grasp_override:
                            # Validate that override task exists in existing tasks
                            obj_base = get_base_object_name(obj_to_grasp)
                            if pattern_grasp_override not in existing_tasks_metadata:
                                raise ValueError(f"task_pattern_overrides grasps_from for '{obj_base}' references task '{pattern_grasp_override}' which is not in existing tasks")
                            
                            # Find the grasped object in the override task
                            override_task_metadata = existing_tasks_metadata[pattern_grasp_override]
                            override_task_steps = override_task_metadata.get('execution_steps', [])
                            
                            # Find the grasped object in the override task (should match by base name)
                            grasp_source = None
                            for step in override_task_steps:
                                stage, obj = parse_execution_step(step)
                                if stage == 'Grasp' and obj:
                                    obj_step_base = get_base_object_name(obj)
                                    if obj_step_base == obj_base:
                                        # Found matching object, use this task and object
                                        grasp_source = (pattern_grasp_override, obj)
                                        break
                            
                            # If no matching object found in override task, fall back to standard logic
                            if grasp_source is None:
                                obj_location = base_parser.initial_states.get(obj_to_grasp) if base_parser.initial_states else None
                                grasp_source = find_grasps_from(
                                    existing_tasks_metadata,
                                    obj_to_grasp,
                                    task_override=pattern_grasp_override,
                                    default_grasp_strategy=default_grasp_strategy,
                                    object_location=obj_location,
                                    place_location=obj_place_location,
                                )
                        else:
                            # Validate that override task exists in existing tasks
                            if final_grasp_override and final_grasp_override not in existing_tasks_metadata:
                                raise ValueError(f"grasps_from override '{final_grasp_override}' for task '{final_filename}' must reference an existing task, but it was not found in existing_tasks_metadata")

                            # Get the location of the object to grasp from initial states
                            obj_location = base_parser.initial_states.get(obj_to_grasp) if base_parser.initial_states else None
                            grasp_source = find_grasps_from(
                                existing_tasks_metadata,
                                obj_to_grasp,
                                task_override=final_grasp_override,
                                default_grasp_strategy=default_grasp_strategy,
                                object_location=obj_location,
                                place_location=obj_place_location,
                            )
                        # Fallback: if no match found with location requirement, try without location
                        if grasp_source is None and obj_location is not None:
                            grasp_source = find_grasps_from(
                                existing_tasks_metadata,
                                obj_to_grasp,
                                task_override=task_override,
                                default_grasp_strategy=default_grasp_strategy,
                                object_location=None,  # Try without location requirement
                                place_location=obj_place_location,
                            )
                        # If we still haven't found a grasp source here, generation
                        # will later fail when demonstrations are created. Make this
                        # explicit so users can add overrides or defaults.
                        if grasp_source is None:
                            raise ValueError(
                                f"Failed to auto-infer grasps_from for object '{obj_to_grasp}' "
                                f"in generated task '{final_filename}'. Please provide a "
                                f"grasps_from override or update default_grasp_strategy/"
                                f"task_pattern_overrides."
                            )
                        if grasp_source is not None:
                            # grasp_source is a tuple of (task_name, object_name)
                            task_name, source_object_name = grasp_source
                            grasps_from[obj_to_grasp] = {
                                'task': task_name,
                                'object': source_object_name
                            }
                
                if grasps_from:
                    cur_metadata['grasps_from'] = grasps_from
                
                # Add is_existing_task flag to metadata
                cur_metadata['is_existing_task'] = is_existing_task
                
                # Check if task should be skipped based on skip_generation_for conditions
                # Only check for newly generated tasks (not existing ones)
                if not is_existing_task and skip_generation_for:
                    execution_steps = cur_metadata.get('execution_steps', [])
                    initial_states = base_parser.initial_states if base_parser.initial_states is not None else None
                    if should_skip_task_generation(
                        chained_task,
                        skip_generation_for,
                        cur_metadata,
                        execution_steps,
                        existing_tasks_metadata,
                        initial_states=initial_states,
                        open_predicates=_parse_open_turnon_from_bddl(bddl_content),
                    ):
                        print(f"Note: Task '{final_filename}' matches skip_generation_for conditions. Skipping generation.")
                        continue
                
                # Store metadata in the appropriate dictionary using final filename
                if is_existing_task:
                    task_metadata_existing[final_filename] = cur_metadata
                else:
                    task_metadata_extra[final_filename] = cur_metadata
                
                # Track the generated task name
                generated_task_names.add(final_filename)
                
                if is_existing_task:
                    # Should not reach here after the logic above, but keep for safety
                    continue
                
                # Save the new bddl file to the extra folder
                out_bddl_dir = out_bddl_dir_extra
                
                # Resize regions for place operations that occur after actions_from steps
                # Parse regions from BDDL
                regions = parse_regions_from_bddl(new_bddl_content)
                
                # Resize regions that are too small (only for steps after actions_from)
                modified_bddl_content, updated_execution_steps, region_name_mapping = resize_regions_for_place_operations(
                    new_bddl_content,
                    execution_steps,
                    num_replayed_steps,
                    min_region_size,
                    regions
                )
                
                # Update execution_steps in metadata if regions were resized
                if region_name_mapping:
                    execution_steps = updated_execution_steps
                    cur_metadata['execution_steps'] = updated_execution_steps
                    new_bddl_content = modified_bddl_content
                    print(f"  Resized {len(region_name_mapping)} region(s) to meet minimum size requirement: {region_name_mapping}")
                
                out_bddl_filename = os.path.join(out_bddl_dir, final_filename + '.bddl')
                Path(out_bddl_filename).parent.mkdir(parents=True, exist_ok=True)
                with Path(out_bddl_filename).open("w", encoding="utf-8") as f:
                    f.write(new_bddl_content)
                
                # Store BDDL content and execution steps for functional equivalence checking against future generated tasks
                # Only store for newly generated tasks (not existing ones)
                if not is_existing_task:
                    generated_task_info[final_filename] = {
                        'bddl_content': new_bddl_content,
                        'execution_steps': execution_steps,
                        'is_chained': is_chained
                    }
        
        # Combine metadata for processing actions_from references
        # Include existing tasks since new tasks may reference them
        # existing_tasks_metadata contains all tasks loaded from BDDL files
        all_task_metadata = {**existing_tasks_metadata, **task_metadata_existing, **task_metadata_extra}
        
        # For the tasks we are pulling actions from include which steps to take
        # Note: actions_from should only reference existing tasks, so we look them up in existing_tasks_metadata
        for task_name, cur_task_metadata in all_task_metadata.items():
            if 'actions_from' in cur_task_metadata:
                actions_from = cur_task_metadata['actions_from']
                cur_task_execution_steps = cur_task_metadata['execution_steps']
                # actions_from should only reference existing tasks
                if actions_from not in existing_tasks_metadata:
                    raise ValueError(f"actions_from '{actions_from}' in task '{task_name}' must reference an existing task, but it was not found in existing_tasks_metadata")
                from_task_metadata = existing_tasks_metadata[actions_from]
                steps_in_from_task = from_task_metadata['execution_steps']
                
                # Compare steps sequentially and include all matching steps
                # Match by exact object instances (since we match by exact instances and init states)
                # For Place steps, also compare the target object by exact instance
                matching_steps = []
                min_len = min(len(cur_task_execution_steps), len(steps_in_from_task))
                
                for i in range(min_len):
                    cur_step = cur_task_execution_steps[i]
                    from_step = steps_in_from_task[i]
                    
                    cur_stage, cur_obj = parse_execution_step(cur_step)
                    from_stage, from_obj = parse_execution_step(from_step)
                    
                    # Steps must have the same stage
                    if cur_stage != from_stage:
                        break
                    
                    # For Place steps, the format is "Place obj1 obj2" where obj2 is the target
                    # We need to parse both objects and compare them by exact instance
                    if cur_stage == 'Place':
                        # Parse both objects from Place step: "Place obj1 obj2"
                        cur_parts = cur_step.split()
                        from_parts = from_step.split()
                        if len(cur_parts) >= 3 and len(from_parts) >= 3:
                            cur_obj1 = cur_parts[1]  # object being placed
                            cur_obj2 = cur_parts[2]  # target location
                            from_obj1 = from_parts[1]
                            from_obj2 = from_parts[2]
                            
                            # Compare by exact instance: both object and target must match exactly
                            if cur_obj1 == from_obj1 and cur_obj2 == from_obj2:
                                matching_steps.append(from_step)
                            else:
                                # Place step doesn't match, stop here
                                break
                        else:
                            # Malformed Place step, stop here
                            break
                    elif cur_obj and from_obj:
                        # For other steps (Grasp, Open, etc.), compare by exact instance
                        if cur_obj == from_obj:
                            matching_steps.append(from_step)
                        else:
                            # Steps don't match, stop here
                            break
                    else:
                        # Steps without objects (shouldn't happen in practice, but handle gracefully)
                        matching_steps.append(from_step)
                
                # At least the first step must match (by exact instance)
                assert len(matching_steps) > 0, \
                    f'The first step of the current task must match the first step of the task we are pulling actions from. ' \
                    f'Current first: {cur_task_execution_steps[0]}, From first: {steps_in_from_task[0]}'
                
                # Verify that the first step matches exactly (same stage and same object instance)
                cur_first_step = cur_task_execution_steps[0]
                from_first_step = steps_in_from_task[0]
                cur_stage, cur_obj = parse_execution_step(cur_first_step)
                from_stage, from_obj = parse_execution_step(from_first_step)
                assert cur_stage == from_stage and cur_obj == from_obj, \
                    f'The first step must match exactly. Current: {cur_first_step} (stage={cur_stage}, obj={cur_obj}), ' \
                    f'From: {from_first_step} (stage={from_stage}, obj={from_obj})'
                
                cur_task_metadata['actions_from_steps'] = matching_steps
    
        # Save the task metadata for existing tasks in the original split directory
        # Don't include grasps_from for existing tasks (only needed for new splits)
        if task_metadata_existing:
            filtered_existing = {
                task_name: {k: v for k, v in task_meta.items() if k != 'grasps_from'}
                for task_name, task_meta in task_metadata_existing.items()
            }
            
            # If --only-tasks is specified, load existing metadata and merge
            if only_tasks_set is not None:
                metadata_path = os.path.join(out_bddl_dir_original, 'task_metadata.yaml')
                existing_metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_metadata = yaml.safe_load(f) or {}
                    existing_tasks = existing_metadata.get('tasks', {})
                    # Merge: keep existing tasks, only update tasks in only_tasks_set
                    merged_tasks = existing_tasks.copy()
                    for task_name, task_meta in filtered_existing.items():
                        if task_name in only_tasks_set:
                            merged_tasks[task_name] = task_meta
                    filtered_existing = merged_tasks
            
            metadata_to_save = {
                'base_split_name': split_name,
                'tasks': filtered_existing
            }
            Path(out_bddl_dir_original).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(out_bddl_dir_original, 'task_metadata.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
    
        # Before saving, enrich second-step-only extra tasks with grasps_from
        # inherited from their originating full chained tasks. This ensures that
        # both the extra split and any views have consistent grasp strategies.
        if task_metadata_extra:
            for extra_task_name, extra_meta in task_metadata_extra.items():
                if not (
                    extra_meta.get('derived_from_chain', False)
                    and extra_meta.get('derived_chain_step', None) == 2
                ):
                    continue
                full_chain_task_name = extra_meta.get('derived_from_full_chain_task')
                if not full_chain_task_name:
                    continue
                # Look for the originating full chained task in extra, existing, or original metadata
                full_chain_meta = (
                    task_metadata_extra.get(full_chain_task_name)
                    or task_metadata_existing.get(full_chain_task_name)
                    or existing_tasks_metadata.get(full_chain_task_name)
                )
                if not full_chain_meta:
                    # Open→PlaceIn chains on the same container are skipped (duplicate of In tasks).
                    # Fall back to the equivalent In task, identified by its execution_steps matching
                    # [Open container, Grasp obj, Place obj container] where Grasp+Place come from
                    # this second-step task's execution_steps.
                    chain_types = extra_meta.get('derived_full_chain_types', [])
                    if chain_types == ['Open', 'PlaceIn']:
                        second_step_steps = extra_meta.get('execution_steps', [])
                        # Reconstruct the Open step from the Place step's target (the container).
                        # e.g. "Place cream_cheese_1 wooden_cabinet_1_top_region" → "Open wooden_cabinet_1_top_region"
                        place_step = next(
                            (s for s in second_step_steps if s.startswith('Place ')), None
                        )
                        open_step = None
                        if place_step:
                            parts = place_step.split()
                            if len(parts) >= 3:
                                open_step = f"Open {parts[2]}"
                        if open_step:
                            expected_in_steps = [open_step] + second_step_steps
                            for search_pool in (task_metadata_extra, task_metadata_existing, existing_tasks_metadata):
                                for candidate_name, candidate_meta in search_pool.items():
                                    if candidate_meta.get('execution_steps') == expected_in_steps:
                                        full_chain_meta = candidate_meta
                                        # For existing tasks that are their own grasp source, inject
                                        # a synthetic grasps_from so the inheritance logic can proceed.
                                        if not full_chain_meta.get('grasps_from') and not full_chain_meta.get('actions_from'):
                                            inferred = {}
                                            for step in expected_in_steps:
                                                stage, obj = parse_execution_step(step)
                                                if stage == 'Grasp' and obj:
                                                    inferred[obj] = {'task': candidate_name, 'object': obj}
                                            if inferred:
                                                full_chain_meta = dict(full_chain_meta, grasps_from=inferred)
                                        break
                                if full_chain_meta:
                                    break
                    if not full_chain_meta:
                        continue
                full_chain_grasps_from = full_chain_meta.get('grasps_from', {})
                if not full_chain_grasps_from:
                    # Full chain's grasp may be covered by actions_from;
                    # use the actions_from task itself as the grasp source.
                    actions_from_task_name = full_chain_meta.get('actions_from')
                    if actions_from_task_name:
                        actions_from_meta = (
                            task_metadata_extra.get(actions_from_task_name)
                            or task_metadata_existing.get(actions_from_task_name)
                            or existing_tasks_metadata.get(actions_from_task_name)
                        )
                        if actions_from_meta is not None:
                            inferred = {}
                            for step in actions_from_meta.get('execution_steps', []):
                                stage, obj = parse_execution_step(step)
                                if stage == 'Grasp' and obj:
                                    inferred[obj] = {'task': actions_from_task_name, 'object': obj}
                            full_chain_grasps_from = inferred
                if not full_chain_grasps_from:
                    continue
                # Collect objects actually grasped in this second-step task
                grasped_objects = set()
                for step in extra_meta.get('execution_steps', []):
                    stage, obj = parse_execution_step(step)
                    if stage == 'Grasp' and obj:
                        grasped_objects.add(obj)
                if not grasped_objects:
                    continue
                inherited_grasps = {
                    obj: info
                    for obj, info in full_chain_grasps_from.items()
                    if obj in grasped_objects
                }
                if not inherited_grasps:
                    continue
                # Merge with any existing grasps_from (do not overwrite explicit entries)
                existing_grasps = extra_meta.get('grasps_from', {})
                merged_grasps = dict(existing_grasps)
                for obj, info in inherited_grasps.items():
                    if obj not in merged_grasps:
                        merged_grasps[obj] = info
                if merged_grasps:
                    extra_meta['grasps_from'] = merged_grasps

        # Save the task metadata for extra tasks (only new tasks)
        if task_metadata_extra:
            # If --only-tasks is specified, load existing metadata and merge
            if only_tasks_set is not None:
                metadata_path = os.path.join(out_bddl_dir_extra, 'task_metadata.yaml')
                existing_metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        existing_metadata = yaml.safe_load(f) or {}
                    existing_tasks = existing_metadata.get('tasks', {})
                    # Merge: keep existing tasks, only update tasks in only_tasks_set
                    merged_tasks = existing_tasks.copy()
                    for task_name, task_meta in task_metadata_extra.items():
                        if task_name in only_tasks_set:
                            merged_tasks[task_name] = task_meta
                    task_metadata_extra = merged_tasks
            
            metadata_to_save = {
                'base_split_name': split_name,
                'tasks': task_metadata_extra
            }
            Path(out_bddl_dir_extra).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(out_bddl_dir_extra, 'task_metadata.yaml'), 'w', encoding='utf-8') as f:
                yaml.dump(metadata_to_save, f, Dumper=NoAliasDumper, default_flow_style=False)
                
        
        # Process views for this split (copy BDDL files and create filtered metadata)
        views_config = split_config.get('views', {})
        if views_config:
            # Load metadata from files (makes view generation independent from BDDL generation)
            task_metadata_by_dir, bddl_dirs = load_metadata_from_files(
                split_name,
                bddl_files_base,
                suffix,
                views_config  # Pass views_config to exclude view directories
            )
            
            # Process views
            view_matching_tasks = process_views_for_split_bddl(
                split_name,
                views_config,
                task_metadata_by_dir,
                bddl_dirs,
                bddl_files_base,
                init_states_output_base,
                suffix,
                gen_init_states
            )
            all_split_view_matching_tasks[split_name] = {
                'view_matching_tasks': view_matching_tasks,
                'views_config': views_config
            }
        
        # Collect individual BDDL files that need init states generated (don't generate yet)
        if gen_init_states:
            if task_metadata_extra:
                # Find all BDDL files in the extra directory
                bddl_files_extra = list(Path(out_bddl_dir_extra).glob("*.bddl"))
                for bddl_file in bddl_files_extra:
                    init_state_tasks.append((str(bddl_file), out_init_states_dir_extra, overwrite_init_states))
        
        # Skip this assertion when --only-tasks is specified, as we're only generating a subset
        if only_tasks_set is None:
            assert existing_tasks_filename_matches == existing_tasks_filenames, f'All of the original tasks must be represented in the set of potential new tasks. Missing: {existing_tasks_filenames - existing_tasks_filename_matches}, Extra: {existing_tasks_filename_matches - existing_tasks_filenames}'
        
        # Track task counts for this split
        num_existing = len(task_metadata_existing)
        expected_existing = len(existing_tasks_filenames)
        # Skip this check when --only-tasks is specified, as we're only generating a subset
        if only_tasks_set is None and num_existing != expected_existing:
            raise ValueError(
                f"Number of existing tasks ({num_existing}) does not match expected number ({expected_existing}) "
                f"for split '{split_name}'. "
                f"Found tasks: {set(task_metadata_existing.keys())}, "
                f"Expected tasks: {existing_tasks_filenames}"
            )
        num_extra = len(task_metadata_extra)
        num_total = num_existing + num_extra
        split_task_counts[split_name] = {
            'existing': num_existing,
            'extra': num_extra,
            'total': num_total
        }
    
    # Generate all init states in parallel after all YAMLs are generated
    # Parallelism is at the individual BDDL file level
    if gen_init_states and init_state_tasks:
        # If overwrite_init_states is True, delete the entire init state directories before generating
        # BUT: if --only-tasks is specified, don't delete directories (only overwrite specific files)
        if overwrite_init_states and only_tasks is None:
            # Extract unique init state directories from tasks
            unique_init_state_dirs = set()
            for _, init_states_dir, _ in init_state_tasks:
                unique_init_state_dirs.add(init_states_dir)
            
            # Delete each directory if it exists
            for init_states_dir in unique_init_state_dirs:
                if os.path.exists(init_states_dir):
                    print(f"Deleting init state directory (overwrite=True): {init_states_dir}")
                    shutil.rmtree(init_states_dir)
        elif overwrite_init_states and only_tasks is not None:
            print(f"Note: --only-tasks specified, will only overwrite specific init files (not deleting directories)")
        
        print(f"\nGenerating init states for {len(init_state_tasks)} tasks in parallel (workers: {num_workers_init_states})...")
        
        def generate_init_states_wrapper(args_tuple):
            bddl_file_path, init_states_dir, overwrite = args_tuple
            try:
                generate_init_states_for_single_bddl(
                    bddl_file_path, 
                    init_states_dir, 
                    overwrite=overwrite,
                    show_progress=True,
                )
                return (bddl_file_path, True, None)
            except Exception as e:
                return (bddl_file_path, False, str(e))
            
        errors_occurred = False
        if num_workers_init_states > 1 and len(init_state_tasks) > 1:
            # Use parallel processing at the task level
            # TODO: maybe move to multi process instead of multi thread
            with ThreadPoolExecutor(max_workers=num_workers_init_states) as executor:
                futures = {executor.submit(generate_init_states_wrapper, task): task 
                            for task in init_state_tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating init states"):
                    bddl_file_path, success, error = future.result()
                    if not success:
                        errors_occurred = True
                        print(f"Error generating init states for {bddl_file_path}: {error}")
        else:
            # Sequential processing
            for task in tqdm(init_state_tasks, desc="Generating init states"):
                bddl_file_path, init_states_dir, overwrite = task
                bddl_file_path, success, error = generate_init_states_wrapper(task)
                if not success:
                    errors_occurred = True
                    print(f"Error generating init states for {bddl_file_path}: {error}")
        
        if errors_occurred:
            print("\nSome init states failed to generate. Please check the errors above.")
        else:
            print("\nAll init states generated successfully!")
    
    # Copy init states for views (after all init states are generated)
    if all_split_view_matching_tasks:
        print("\n" + "="*80)
        print("COPYING INIT STATES FOR VIEWS")
        print("="*80)
        for split_name, view_data in all_split_view_matching_tasks.items():
            view_matching_tasks = view_data['view_matching_tasks']
            views_config = view_data['views_config']
            
            # Build init_states_dirs dict for this split
            # For existing tasks, use default (non-suffixed) path if suffix is provided
            # because existing tasks' init states are not copied to suffixed directories
            default_init_states_base = get_libero_path('init_states')
            if suffix is not None:
                init_states_dirs = {
                    'existing': os.path.join(default_init_states_base, split_name),
                    'extra': os.path.join(init_states_output_base, f'{split_name}_extra')
                }
            else:
                init_states_dirs = {
                    'existing': os.path.join(init_states_output_base, split_name),
                    'extra': os.path.join(init_states_output_base, f'{split_name}_extra')
                }
            
            copy_init_states_for_views(
                split_name,
                views_config,
                view_matching_tasks,
                init_states_dirs,
                init_states_output_base,
                suffix,
                gen_init_states
            )
        print("="*80 + "\n")
    
    # Print summary of tasks created per split
    print("\n" + "="*80)
    print("TASK GENERATION SUMMARY")
    print("="*80)
    total_existing = 0
    total_extra = 0
    total_all = 0
    
    for split_name, counts in split_task_counts.items():
        print(f"\n{split_name}:")
        print(f"  Existing tasks: {counts['existing']}")
        print(f"  Extra tasks:    {counts['extra']}")
        print(f"  Total tasks:    {counts['total']}")
        total_existing += counts['existing']
        total_extra += counts['extra']
        total_all += counts['total']
    
    print("\n" + "-"*80)
    print("TOTALS (across all splits):")
    print(f"  Existing tasks: {total_existing}")
    print(f"  Extra tasks:    {total_extra}")
    print(f"  Total tasks:    {total_all}")
    print("="*80)
    
    # Print summary of views generated per split
    if all_split_view_matching_tasks:
        print("\n" + "="*80)
        print("VIEWS GENERATION SUMMARY")
        print("="*80)
        
        total_view_tasks = 0
        total_views = 0
        
        for split_name, view_data in all_split_view_matching_tasks.items():
            view_matching_tasks = view_data['view_matching_tasks']
            views_config = view_data['views_config']
            
            if not view_matching_tasks:
                continue
            
            print(f"\n{split_name}:")
            for view_name, matching_tasks in view_matching_tasks.items():
                view_count = len(matching_tasks)
                print(f"  {view_name}: {view_count} tasks")
                total_view_tasks += view_count
                total_views += 1
        
        if total_views > 0:
            print("\n" + "-"*80)
            print("TOTALS (across all splits):")
            print(f"  Total views:    {total_views}")
            print(f"  Total view tasks: {total_view_tasks}")
            print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-envs-config-path", type=str, default='gen_extra_libero_envs_configs/tasks_spec.yaml')
    parser.add_argument("--overwrite-init-states", action="store_true", dest="overwrite_init_states")
    parser.add_argument("--no-overwrite-init-states", action="store_true", dest="no_overwrite_init_states")
    parser.add_argument("--skip-gen-init-states", action="store_true", help="Skip init state generation (init states are generated by default)")
    parser.add_argument("--num-workers-init-states", type=int, default=40, help="Number of worker threads for init state generation (default: 40)")
    parser.add_argument("--splits", type=str, nargs="+", default=None, help="List of splits to process (e.g., --splits libero_goal libero_spatial). If not specified, all splits are processed.")
    parser.add_argument("--only-tasks", type=str, nargs="+", default=None, help="List of task names to regenerate. If specified, only these tasks will be generated and existing BDDLs/init states won't be deleted at the start.")
    parser.add_argument("--suffix", type=str, default=None, help="Suffix to append to bddl_files and init_states base paths (e.g., 'test' will make paths like 'bddl_files_test' and 'init_states_test'). Underscore is added automatically.")
    parser.add_argument("--min-region-size", type=float, default=0.0064, help="Minimum area size required for regions used in place operations (default: 0.0064)")
    parser.add_argument("--just-views", action="store_true", help="Only generate views from existing BDDL files and metadata, don't regenerate BDDLs or init states")
    args = parser.parse_args()

    # Validate that both flags are not provided simultaneously
    if args.overwrite_init_states and args.no_overwrite_init_states:
        parser.error("Cannot specify both --overwrite-init-states and --no-overwrite-init-states")

    start_time = time.time()
    
    if args.just_views:
        # Only generate views from existing files
        generate_views_only(args.extra_envs_config_path, args.splits, args.suffix)
    else:
        # Normal flow with BDDL generation
        # Invert the logic: if --skip-gen-init-states is passed, gen_init_states = False, otherwise True (default)
        gen_init_states = not args.skip_gen_init_states
        gen_extra_libero_envs(args.extra_envs_config_path, args.overwrite_init_states, args.no_overwrite_init_states, gen_init_states, args.num_workers_init_states, args.splits, args.only_tasks, args.suffix, args.min_region_size)
    
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/3600} hours")
