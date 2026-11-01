def algo_label(target: str) -> str:
    """Derive results filename stem from ga._target_ (e.g. StandardGA -> standard)."""
    if target.endswith(".ga.ImprovedGA"):
        return "ga"
    name = target.rsplit(".", 1)[-1]
    return name.removesuffix("GA").lower()
