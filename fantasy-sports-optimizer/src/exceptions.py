"""
Shared custom exception types for the Fantasy Sports Team Optimizer.

All exceptions defined here are imported by the relevant modules
(model.py, optimizer.py, serialization.py) so that callers only need
to import from a single, stable location.
"""


class ModelNotTrainedError(Exception):
    """Raised when :py:meth:`FantasyModel.predict` is called before
    :py:meth:`FantasyModel.train` has been invoked.

    Example::

        model = FantasyModel()
        model.predict(df)  # raises ModelNotTrainedError
    """


class InvalidInputError(ValueError):
    """Raised by :py:class:`TeamOptimizer` when the ``predicted_points``
    array passed to :py:meth:`TeamOptimizer.optimize` contains NaN, Inf,
    or non-numeric values.

    The exception is raised *before* the solver is invoked so that the
    caller receives immediate feedback without waiting for a solver timeout.

    Example::

        optimizer = TeamOptimizer()
        optimizer.optimize(players_df, np.array([float("nan"), 5.0]))
        # raises InvalidInputError
    """


class InfeasibleRosterError(Exception):
    """Raised by :py:class:`TeamOptimizer` when the integer program solver
    definitively determines that no feasible 11-player roster exists under
    the given constraints.

    The ``constraint_class`` attribute (str) names the specific constraint
    category that caused infeasibility (e.g. ``"budget"``,
    ``"position_limit"``, ``"team_cap"``).

    The exception is raised *only after* the solver has confirmed
    infeasibility — it is never raised while the solver is still searching.

    Example::

        optimizer = TeamOptimizer(budget=1.0)  # impossibly tight budget
        optimizer.optimize(players_df, predicted_points)
        # raises InfeasibleRosterError("No feasible roster: budget constraint")
    """

    def __init__(self, message: str, constraint_class: str = "unknown") -> None:
        """Initialise the exception.

        Args:
            message: Human-readable description of the infeasibility.
            constraint_class: The constraint category that caused
                infeasibility.  Defaults to ``"unknown"``.
        """
        super().__init__(message)
        self.constraint_class = constraint_class


class RosterDeserializationError(ValueError):
    """Raised by :py:func:`roster_from_json` when the supplied JSON string
    cannot be parsed or does not conform to the expected roster schema.

    Two distinct sub-cases are covered by this single exception type:

    1. **Invalid JSON** — the string is not valid JSON at all.
       The message will contain the phrase ``"JSON parsing failed"``.

    2. **Schema violation** — the JSON is valid but a required field is
       missing or has the wrong type.  The message will identify the
       offending field name and the expected Python type.

    Example::

        roster_from_json("not json")
        # raises RosterDeserializationError("JSON parsing failed: ...")

        roster_from_json('[{"player_id": "oops"}]')
        # raises RosterDeserializationError(
        #     "Field 'player_id' expected type int, got str"
        # )
    """
