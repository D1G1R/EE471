"""
Programming for the Puzzled -- Srini Devadas
You Will All Conform

Input is a list of 'F's and 'B's, in terms of forwards and backwards caps.
Output is a set of commands (printed out) to get either all 'F's or all 'B's.
Fewest commands are the goal.
"""

from typing import Literal

CapDirection = Literal["F", "B"]
FORWARD: CapDirection = "F"
BACKWARD: CapDirection = "B"


def find_intervals(caps: list[CapDirection]) -> list[tuple[int, int, CapDirection]]:
    """
    Find contiguous intervals where caps are in the same direction.

    Args:
        caps: A list of cap directions ('F' for forward, 'B' for backward).

    Returns:
        A list of tuples, each containing (start_index, end_index, direction).
    """
    if not caps:
        return []

    intervals: list[tuple[int, int, CapDirection]] = []
    start = 0

    for i in range(1, len(caps)):
        if caps[start] != caps[i]:
            intervals.append((start, i - 1, caps[start]))
            start = i

    # Add the last interval after the loop completes
    intervals.append((start, len(caps) - 1, caps[start]))

    return intervals


def determine_flip_target(
    intervals: list[tuple[int, int, CapDirection]],
) -> CapDirection:
    """
    Determine which cap direction requires fewer flips.

    Args:
        intervals: List of intervals with their cap directions.

    Returns:
        The cap direction ('F' or 'B') that appears fewer times.
    """
    forward_count = sum(1 for _, _, cap in intervals if cap == FORWARD)
    backward_count = len(intervals) - forward_count

    return FORWARD if forward_count < backward_count else BACKWARD


def generate_commands(
    intervals: list[tuple[int, int, CapDirection]], flip_target: CapDirection
) -> list[str]:
    """
    Generate flip commands for the target intervals.

    Args:
        intervals: List of intervals with their cap directions.
        flip_target: The cap direction to flip to.

    Returns:
        A list of command strings.
    """
    commands: list[str] = []

    for start_idx, end_idx, cap_type in intervals:
        if cap_type == flip_target:
            if start_idx == end_idx:
                commands.append(f"Person in position {start_idx} flip your cap!")
            else:
                commands.append(
                    f"People in positions {start_idx} through {end_idx} flip your caps!"
                )

    return commands


def please_conform(caps: list[CapDirection]) -> list[str]:
    """
    Determine the minimal commands needed to make all caps conform.

    Analyzes the input list and returns commands to flip the minimum
    number of caps so that all caps face the same direction.

    Args:
        caps: A list of cap directions ('F' for forward, 'B' for backward).

    Returns:
        A list of command strings indicating which positions should flip.
        Returns an empty list if the input is empty or already uniform.
    """
    if not caps:
        return []

    intervals = find_intervals(caps)
    flip_target = determine_flip_target(intervals)

    return generate_commands(intervals, flip_target)


def display_commands(commands: list[str]) -> None:
    """
    Print the given commands to standard output.

    Args:
        commands: A list of command strings to display.
    """
    for command in commands:
        print(command)


if __name__ == "__main__":
    caps_list = ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'B', 'F']
    caps_list_2 = ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'F', 'F']

    print("Testing first caps list:")
    commands = please_conform(caps_list)
    display_commands(commands)

    print("\nTesting second caps list:")
    commands = please_conform(caps_list_2)
    display_commands(commands)
