"""
Programming for the Puzzled -- Srini Devadas
You Will All Conform

Input is a list of 'F's and 'B's, in terms of forwards and backwards caps.
Output is a set of commands (printed out) to get either all 'F's or all 'B's.
Fewest commands are the goal.
"""

def please_conform(caps: list[str]) -> None:
    """
    Determines and prints the minimal number of commands needed 
    to make all elements in the array conform to the same direction.
    """
    if not caps:
        return

    # Initialization
    start = 0
    forward_count: int = 0
    backward_count: int = 0
    intervals = []

    # Determine intervals where caps are in the same direction
    for i in range(1, len(caps)):
        if caps[start] != caps[i]:
            # each interval is a tuple with 3 elements: (start, end, type)
            current_cap = caps[start]
            intervals.append((start, i - 1, current_cap))
            
            if current_cap == 'F':
                forward_count += 1
            else:
                backward_count += 1
                
            start = i

    # Need to add the last interval after the loop completes execution
    current_cap = caps[start]
    intervals.append((start, len(caps) - 1, current_cap))
    
    if current_cap == 'F':
        forward_count += 1
    else:
        backward_count += 1

    # Determine which cap direction requires fewer intervals to flip
    flip_target = 'F' if forward_count < backward_count else 'B'

    # Print out the commands
    for start_idx, end_idx, cap_type in intervals:
        if cap_type == flip_target:
            if start_idx == end_idx:
                print(f"Person in position {start_idx} flip your cap!")
            else:
                print(f"People in positions {start_idx} through {end_idx} flip your caps!")


if __name__ == "__main__":
    caps_list = ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'B', 'F']
    caps_list_2 = ['F', 'F', 'B', 'B', 'B', 'F', 'B', 'B', 'B', 'F', 'F', 'F', 'F']
    
    print("Testing first caps list:")
    please_conform(caps_list)
    
    # print("\nTesting second caps list:")
    # please_conform(caps_list_2)
