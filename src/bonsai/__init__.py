"""Topological--Riemannian BONSAI components."""

from src.bonsai.adapters import SharedLowRankAdapter
from src.bonsai.atgtr import AdaptiveTaskGraphTrustRegion
from src.bonsai.geometry import LowRankRiemannianMetric
from src.bonsai.hierarchy import TaskHierarchy
from src.bonsai.model import BONSAIModel, BONSAIModelOutput
from src.bonsai.ot import SlicedWasserstein, TaskDistributionDescriptor
from src.bonsai.repository import TaskRecord, TaskRepository
from src.bonsai.replay import AdaptiveTaskGraphFunctionalReplay, FunctionalMemory
from src.bonsai.rgsc import SubspaceAnchor, TopologyGatedRiemannianSubspaceConsolidator
from src.bonsai.router import RouteResult, TaskRouter
from src.bonsai.sheaf import SparseTaskSheaf
from src.bonsai.system import BONSAISystem
from src.bonsai.tda import PersistenceDescriptor, ZeroDimensionalPersistence
from src.bonsai.vib import VIBEncoder, VIBOutput

__all__ = [
    "AdaptiveTaskGraphTrustRegion", "BONSAIModel", "BONSAIModelOutput", "BONSAISystem", "LowRankRiemannianMetric",
    "PersistenceDescriptor", "RouteResult", "SharedLowRankAdapter", "SlicedWasserstein",
    "SparseTaskSheaf", "TaskDistributionDescriptor", "TaskHierarchy", "TaskRecord",
    "TaskRepository", "TaskRouter", "AdaptiveTaskGraphFunctionalReplay", "FunctionalMemory",
    "TopologyGatedRiemannianSubspaceConsolidator",
    "SubspaceAnchor", "VIBEncoder", "VIBOutput", "ZeroDimensionalPersistence",
]
