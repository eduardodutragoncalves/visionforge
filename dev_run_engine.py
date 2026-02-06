import sys
import os
import uvicorn

# Ensure engine package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting VisionForge Engine Dev Server...")
    uvicorn.run(
        "engine.main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )
