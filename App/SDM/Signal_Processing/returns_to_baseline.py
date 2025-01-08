def returns_to_baseline(previous_avg: float, tac_list_current: list, index: int, change_downward: bool,
                        max_distance: int = 120) -> bool:
    back_index = min(len(tac_list_current) - 1, index + max_distance)
    comparison = (lambda tac: tac < previous_avg) if not change_downward else (lambda tac: tac > previous_avg)
    return any(comparison(tac) for tac in tac_list_current[index:back_index])
