from fastapi import APIRouter, Request
from datetime import datetime

router = APIRouter()

@router.get("/health")
def health(request: Request):
    """Laravel appelle cet endpoint pour vérifier que le service ML est actif."""
    models = request.app.state.models
    return {
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "pricing":  {
                "loaded":   True,
                "metrics":  models["pricing"]["metrics"],
            },
            "matching": {
                "loaded":   True,
                "metrics":  models["matching"]["metrics"],
            },
        }
    }
