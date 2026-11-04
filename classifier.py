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

    def classify(self, roi, model_result: dict):
        self.product_count += 1

        algo = model_result.get("algo", "draem")
        defect = model_result.get("defect", False)
        score = model_result.get("score", 0)
        defects = [algo] if defect else []

        if defect:
            self.ng_count += 1
            verdict = "NG"
        else:
            self.good_count += 1
            verdict = "GOOD"

        logger.info(f"Product #{self.product_count}: {verdict} ({algo} score={score:.3f})")

        return InspectionResult(
            product_id=self.product_count,
            timestamp=time.time(),
            has_defect=defect,
            score=round(score, 4),
            details={
                "verdict": verdict,
                "defects_found": defects,
                "algorithm_results": {
                    algo: {"defect": defect, "score": score},
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
