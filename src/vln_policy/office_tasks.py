"""Public language tasks. No scene coordinates, asset names, or oracle data."""
TASKS = {
    "meeting": {
        "instruction": "Explore this unfamiliar office. Find a meeting area containing a shared meeting table, multiple chairs, and a large wall-mounted presentation display. Enter its free public floor area, verify the combination from a second viewpoint, and stop. Individual workstation desks with small PC monitors, ceiling light panels, windows, and lounge coffee tables do not satisfy this task.",
        "features": ["shared_meeting_table", "multiple_chairs", "wall_display"],
        "criteria": "one shared conference table, multiple chairs around it, and a large wall-mounted presentation screen",
        "search_hint": "A lounge is not the target. Investigate open doorways and partially visible meeting furniture before continuing down corridors.",
        "arrival": "The camera must be inside the meeting room on free floor next to the shared table, not outside a doorway or glass partition.",
    },
    "lounge": {
        "instruction": "Explore this unfamiliar office. Find a waiting lounge with a large wall-mounted display, a group of at least two upholstered armchairs, and a small table. Enter the free floor next to that seating group, verify the combination from a second viewpoint, and stop. Workstation monitors, rolling office chairs, ceiling lights, and a reception counter alone do not satisfy the task.",
        "features": ["armchair_group", "small_table", "wall_display"],
        "criteria": "at least two upholstered armchairs grouped together, a small table beside or between them, and a large wall-mounted screen nearby",
        "search_hint": "Inspect grouped upholstered seats and large wall-mounted screens; reject workstations and reception counters without the seating group.",
        "arrival": "This is an open waiting lounge, so no doorway or enclosed boundary has to be crossed. Standing on its open floor roughly one to three metres in front of the armchair group counts as being in the area. Do not require the robot to stand between the chairs or touch the furniture.",
    },
}
