"""
Firebase Client for State Management and Real-time Data Streaming
Implements robust error handling, connection pooling, and automatic retry logic.
"""
import logging
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from google.cloud import firestore
from google.cloud.firestore_v1.client import Client as FirestoreClient
from google.cloud.firestore_v1.document import DocumentReference
from google.cloud.firestore_v1.collection import CollectionReference
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FirebaseClient:
    """Robust Firebase client with automatic reconnection and error handling"""
    
    def __init__(self):
        self._client: Optional[FirestoreClient] = None
        self._connected: bool = False
        self._connection_attempts: