"""
Pytest configuration and shared Hypothesis settings for the
Fantasy Sports Team Optimizer test suite.

Run the full suite with::

    pytest tests/ -v --hypothesis-seed=0
"""

from hypothesis import settings, HealthCheck

# Apply a profile that is suitable for the full test suite:
#   - max_examples=100  : enough coverage without being too slow
#   - deadline=None     : optimizer and model training calls can be slow;
#                         disabling the deadline prevents spurious failures
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.load_profile("ci")
