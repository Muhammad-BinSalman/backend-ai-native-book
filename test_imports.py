import sys
import os
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    from app.config.settings import settings
    print("Settings imported successfully")
except ImportError as e:
    print(f"Failed to import settings: {e}")

try:
    from scripts.ingest_book import BookIngestionOrchestrator
    print("Orchestrator imported successfully")
except ImportError as e:
    print(f"Failed to import orchestrator: {e}")
