'''Library to support producing synthetic RVS instructions.'''

def describe_route(pivot_poi: str, goal_poi: str) -> str:
  '''Preliminary example template for generating an RVS instruction.
  
  Arguments:
    pivot_poi: The POI used to orient with respect to the goal.
    goal_poi: The POI that is the intended meeting location.
  Returns:
    A string describing the goal location with respect to the reference.
  
  '''
  return f'go to the {goal_poi} near the {pivot_poi}.'