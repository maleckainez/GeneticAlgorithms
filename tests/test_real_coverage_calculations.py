"""Temporary imports for coverage calculations."""


def test_all_core_modules_import():
    import src.classes.ChildrenHandler  # noqa: F401
    import src.classes.EvolutionRunner  # noqa: F401
    import src.classes.ExperimentConfig  # noqa: F401
    import src.classes.OutputGenerator  # noqa: F401
    import src.classes.PathResolver  # noqa: F401
    import src.classes.Plotter  # noqa: F401
    import src.classes.PopulationHandler  # noqa: F401
    import src.classes.Reproduction  # noqa: F401
    import src.methods.experiment_defining_tools  # noqa: F401
    import src.methods.fitness_score  # noqa: F401
    import src.methods.logging_library  # noqa: F401
    import src.methods.selection_methods  # noqa: F401
