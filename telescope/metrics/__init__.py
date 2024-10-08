from .sacrebleu import sacreBLEU
from .chrf import chrF
from .zero_edit import ZeroEdit

# from .bleurt import BLEURT
from .bertscore import BERTScore
from .comet import COMET
from .doc_comet import DocCOMET
from .ter import TER
# from .prism import Prism
from .gleu import GLEU
from .result import MetricResult, PairwiseResult, BootstrapResult


AVAILABLE_METRICS = [
    COMET,
    DocCOMET,
    sacreBLEU,
    chrF,
    ZeroEdit,
    # BLEURT,
    BERTScore,
    TER,
    # Prism,
    GLEU,
]
