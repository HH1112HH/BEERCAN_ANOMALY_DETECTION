import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class InspectionResult:
    product_id: int
    timestamp: float
    has_defect: bool
    score: float
    details: dict = field(default_factory=dict)
    image: any = None

class Classifier:
    def __init__(self):
        self.product_count = 0
        self.good_count = 0
        self.ng_count = 0

    def classify(self, roi, patchcore_result: dict):
        self.product_count += 1

        pc_defect = patchcore_result.get("defect", False)
        pc_score = patchcore_result.get("score", 0)
        defects = ["patchcore"] if pc_defect else []

        if pc_defect:
            self.ng_count += 1
            verdict = "NG"
        else:
            self.good_count += 1
            verdict = "GOOD"

        logger.info(f"Product #{self.product_count}: {verdict} (score={pc_score:.3f})")

        return InspectionResult(
            product_id=self.product_count,
            timestamp=time.time(),
            has_defect=pc_defect,
            score=round(pc_score, 4),
            details={
                "verdict": verdict,
                "defects_found": defects,
                "algorithm_results": {
                    "patchcore": {"defect": pc_defect, "score": pc_score},
                },
                "good_count": self.good_count,
                "ng_count": self.ng_count,
                "total": self.product_count,
                "rate": round(self.good_count / self.product_count * 100, 1) if self.product_count > 0 else 100,
            },
            image=roi,
        )

    def reset(self):
        self.product_count = 0
        self.good_count = 0
        self.ng_count = 0
