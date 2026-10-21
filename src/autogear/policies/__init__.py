"""Policies compared in the study.

Target-based policies (a target development + the shared controller):
    decision tables R0, R1, RT                      -> policies.tables
    learned target models GBDT, LIN, TAB, MLP, RF   -> policies.targets
Event-based policy (the first approach of AutoGear):
    two-stage boosted classifier of shift events    -> policies.event
"""
from .tables import DecisionTable, KINDS as TABLE_KINDS
from .targets import TargetModel, KINDS as MODEL_KINDS

ALL_TARGET_KINDS = (*TABLE_KINDS, *MODEL_KINDS)


def make(kind: str):
    return DecisionTable(kind) if kind in TABLE_KINDS else TargetModel(kind)
