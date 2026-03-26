"""
Programming for the Puzzled -- Srini Devadas
You Will All Conform (Improved Version)

Input is a list of 'F's and 'B's, in terms of forwards and backwards caps.
Output is a set of commands (generated as strings) to get either all 'F's or all 'B's.
Fewest commands are the goal.

This improved version follows clean coding best practices including:
- Separation of concerns (logic vs presentation)
- Type hints and comprehensive documentation
- Input validation and error handling
- DRY principle (no code duplication)
- Testability (functions return values)
- Constants for magic strings
"""

from typing import NamedTuple
from enum import Enum


# ============================================================================
# Constants and Enums
# ============================================================================

class CapDirection(str, Enum):
    """
    Enumeration for cap directions.
    
    Attributes:
        FORWARD: Cap facing forwards ('F')
        BACKWARD: Cap facing backwards ('B')
    """
    FORWARD = 'F'
    BACKWARD = 'B'


class CapInterval(NamedTuple):
    """
    Named tuple representing a contiguous interval of caps with same direction.
    
    Attributes:
        start_index: Starting position of the interval (inclusive)
        end_index: Ending position of the interval (inclusive)
        direction: Direction of caps in this interval (CapDirection)
    """
    start_index: int
    end_index: int
    direction: CapDirection


# ============================================================================
# Helper Functions
# ============================================================================

def find_cap_intervals(caps: list[str]) -> list[CapInterval]:
    """
    Identifies contiguous intervals where caps face the same direction.
    
    This function scans through the caps list and creates intervals whenever
    the cap direction changes, allowing us to identify groups of consecutive
    caps that need to be flipped together.
    
    Args:
        caps: A list of cap directions ('F' for forward, 'B' for backward)
        
    Returns:
        A list of CapInterval objects representing contiguous groups
        
    Raises:
        ValueError: If caps list is empty or contains invalid values
        
    Examples:
        >>> find_cap_intervals(['F', 'F', 'B', 'B'])
        [CapInterval(0, 1, <CapDirection.FORWARD: 'F'>), 
         CapInterval(2, 3, <CapDirection.BACKWARD: 'B'>)]
    """
    if not caps:
        raise ValueError("Caps list cannot be empty")
    
    # Validate all caps are valid directions
    valid_directions = {direction.value for direction in CapDirection}
    invalid_caps = [cap for cap in caps if cap not in valid_directions]
    if invalid_caps:
        raise ValueError(
            f"Invalid cap directions found: {invalid_caps}. "
            f"Valid directions are: {valid_directions}"
        )
    
    intervals: list[CapInterval] = []
    interval_start = 0
    
    # Scan through caps to find direction changes
    for i in range(1, len(caps)):
        if caps[interval_start] != caps[i]:
            # Found a direction change, save the completed interval
            current_direction = CapDirection(caps[interval_start])
            intervals.append(
                CapInterval(interval_start, i - 1, current_direction)
            )
            interval_start = i
    
    # Add the final interval
    final_direction = CapDirection(caps[interval_start])
    intervals.append(
        CapInterval(interval_start, len(caps) - 1, final_direction)
    )
    
    return intervals


def count_intervals_by_direction(
    intervals: list[CapInterval],
) -> tuple[int, int]:
    """
    Counts intervals grouped by cap direction.
    
    Args:
        intervals: List of CapInterval objects to count
        
    Returns:
        Tuple of (forward_count, backward_count)
    """
    forward_count = sum(
        1 for interval in intervals
        if interval.direction == CapDirection.FORWARD
    )
    backward_count = sum(
        1 for interval in intervals
        if interval.direction == CapDirection.BACKWARD
    )
    return forward_count, backward_count


def determine_flip_direction(
    forward_count: int,
    backward_count: int,
) -> CapDirection:
    """
    Determines which direction requires fewer flips.
    
    The goal is to minimize the number of flip commands, so we choose
    to flip the direction that appears in fewer intervals.
    
    Args:
        forward_count: Number of intervals with caps facing forward
        backward_count: Number of intervals with caps facing backward
        
    Returns:
        The CapDirection that should be flipped (the one with fewer intervals)
    """
    return (
        CapDirection.BACKWARD
        if forward_count > backward_count
        else CapDirection.FORWARD
    )


def generate_flip_commands(
    intervals: list[CapInterval],
    flip_direction: CapDirection,
) -> list[str]:
    """
    Generates human-readable flip commands for intervals that need flipping.
    
    This function creates instructions for each interval that matches the
    flip_direction, providing contextual feedback about position ranges.
    
    Args:
        intervals: List of CapInterval objects to process
        flip_direction: The direction we want to flip
        
    Returns:
        A list of command strings describing which people need to flip
        
    Examples:
        >>> intervals = [CapInterval(0, 0, CapDirection.FORWARD), ...]
        >>> generate_flip_commands(intervals, CapDirection.FORWARD)
        ['Person in position 0 flip your cap!', ...]
    """
    commands: list[str] = []
    
    for interval in intervals:
        if interval.direction == flip_direction:
            if interval.start_index == interval.end_index:
                # Single person needs to flip
                command = (
                    f"Person in position {interval.start_index} "
                    f"flip your cap!"
                )
            else:
                # Multiple people need to flip
                command = (
                    f"People in positions {interval.start_index} "
                    f"through {interval.end_index} flip your caps!"
                )
            commands.append(command)
    
    return commands


# ============================================================================
# Main Function
# ============================================================================

def generate_conform_commands(caps: list[str]) -> list[str]:
    """
    Generates minimal flip commands to make all caps face the same direction.
    
    This is the core algorithm that determines the optimal strategy to make
    all caps conform to a single direction (either all 'F' or all 'B')
    with the minimum number of flip commands.
    
    Algorithm:
        1. Identify contiguous intervals of same direction
        2. Count intervals by direction
        3. Choose to flip the direction with fewer intervals
        4. Generate flip commands for that direction
    
    Args:
        caps: A list of cap directions ('F' for forward, 'B' for backward)
        
    Returns:
        A list of command strings describing the flips needed
        
    Raises:
        ValueError: If caps list is empty or contains invalid values
        
    Examples:
        >>> caps = ['F', 'F', 'B', 'B', 'B']
        >>> commands = generate_conform_commands(caps)
        >>> commands
        ['People in positions 2 through 4 flip your caps!']
    """
    if not caps:
        return []
    
    # Step 1: Find intervals of same direction
    intervals = find_cap_intervals(caps)
    
    # Step 2: Count intervals by direction
    forward_count, backward_count = count_intervals_by_direction(intervals)
    
    # Step 3: Determine which direction to flip
    flip_direction = determine_flip_direction(forward_count, backward_count)
    
    # Step 4: Generate commands
    commands = generate_flip_commands(intervals, flip_direction)
    
    return commands


def print_conform_commands(caps: list[str]) -> None:
    """
    Generates and prints conform commands for the given caps.
    
    This is a convenience wrapper function that generates commands
    and prints them to the console.
    
    Args:
        caps: A list of cap directions ('F' for forward, 'B' for backward)
        
    Raises:
        ValueError: If caps list is invalid
    """
    commands = generate_conform_commands(caps)
    for command in commands:
        print(command)


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'B', 'F'],
        ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'F', 'F'],
        ['F'],
        ['B', 'B', 'B'],
        ['F', 'B', 'F', 'B'],
    ]
    
    for i, caps_list in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {caps_list}")
        print(f"{'='*60}")
        try:
            print_conform_commands(caps_list)
        except ValueError as error:
            print(f"Error: {error}")
