"""IntelliDoc main entry point.

Runs the FastAPI backend application via Uvicorn.
"""

import uvicorn
from backend.config import settings
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )

