from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from fastapi import HTTPException
from typing import List, Dict, Any, Optional, Union
import config
import asyncio
import os
from pathlib import Path
import shutil
import json
import logging
from datetime import datetime
import hashlib

from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, DateTime
from sqlalchemy.sql import select, delete

logger = logging.getLogger(__name__)

# Global DocumentStorage instance
_document_storage = None

def get_document_storage() -> 'DocumentStorage':
    """Get or create a global DocumentStorage instance."""
    global _document_storage
    if _document_storage is None:
        _document_storage = DocumentStorage()
    return _document_storage

def store_documents(
    documents: List[Dict],
    copy_files: bool = True
) -> List[str]:
    """Wrapper function to store documents using DocumentStorage.
    
    Args:
        documents: List of document dictionaries with metadata
        copy_files: Whether to copy the original files to storage
        
    Returns:
        List of document IDs
    """
    storage = get_document_storage()
    return storage.store_documents(documents, copy_files=copy_files)

def retrieve_documents(
    document_ids: Optional[List[str]] = None,
    metadata_filters: Optional[Dict] = None
) -> List[Dict]:
    """Wrapper function to retrieve documents using DocumentStorage.
    
    Args:
        document_ids: List of document IDs to retrieve
        metadata_filters: Metadata filters to apply
        
    Returns:
        List of document dictionaries
    """
    storage = get_document_storage()
    return storage.retrieve_documents(document_ids, metadata_filters=metadata_filters)

def delete_documents(
    document_ids: List[str],
    delete_files: bool = True
) -> List[str]:
    """Wrapper function to delete documents using DocumentStorage.
    
    Args:
        document_ids: List of document IDs to delete
        delete_files: Whether to delete the stored files
        
    Returns:
        List of successfully deleted document IDs
    """
    storage = get_document_storage()
    return storage.delete_documents(document_ids, delete_files=delete_files)

# Initialize Azure Blob Storage clients
try:
    blob_service_client = BlobServiceClient.from_connection_string(
        config.AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client(
        config.AZURE_STORAGE_CONTAINER
    )
    config.logger.info(f"Successfully connected to Azure Blob Storage container: {config.AZURE_STORAGE_CONTAINER}")
except Exception as e:
    config.logger.error(f"Error initializing Azure Blob Storage: {str(e)}")
    raise

def get_blob_client(file_name: str):
    """Helper function to get a block blob client for a specific file"""
    return container_client.get_blob_client(file_name)

async def download_file(file_name: str) -> bytes:
    """Download file from Azure Blob Storage and return its bytes"""
    try:
        config.logger.info(f"Attempting to download file: {file_name}")
        blob_client = get_blob_client(file_name)
        blob_data = blob_client.download_blob()
        content = blob_data.readall()
        config.logger.info(f"Successfully downloaded file: {file_name}")
        return content
    except ResourceNotFoundError:
        config.logger.error(f"File not found in Azure storage: {file_name}")
        raise HTTPException(status_code=404, detail=f"File {file_name} not found in Azure storage")
    except Exception as e:
        config.logger.error(f"Error processing file {file_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

async def upload_file_to_blob(file_name: str, content: bytes) -> bool:
    """Upload file to Azure Blob Storage"""
    try:
        blob_client = get_blob_client(file_name)
        blob_client.upload_blob(content, overwrite=True)
        config.logger.info(f"Successfully uploaded file to Azure: {file_name}")
        return True
    except Exception as e:
        config.logger.error(f"Error uploading file to Azure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

def list_files() -> List[Dict[str, Any]]:
    """List all files in the Azure Blob Storage container"""
    try:
        files = []
        for blob in container_client.list_blobs():
            files.append({
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat()
            })
        return files
    except Exception as e:
        config.logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

def delete_file(file_name: str) -> bool:
    """Delete a file from Azure Blob Storage"""
    try:
        blob_client = get_blob_client(file_name)
        blob_client.delete_blob()
        config.logger.info(f"Successfully deleted file from Azure: {file_name}")
        return True
    except ResourceNotFoundError:
        config.logger.warning(f"File not found when attempting to delete: {file_name}")
        return False
    except Exception as e:
        config.logger.error(f"Error deleting file from Azure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

class DocumentStorage:
    def __init__(
        self,
        connection_string: Optional[str] = None,
        storage_dir: str = "./storage",
        table_name: str = "documents"
    ):
        """Initialize document storage.
        
        Args:
            connection_string: PostgreSQL connection string
            storage_dir: Directory for storing document files
            table_name: Name of the documents table
        """
        # Set up database connection
        if connection_string is None:
            connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
            if connection_string is None:
                raise ValueError("PostgreSQL connection string not provided")
                
        self.engine = create_engine(connection_string)
        self.metadata = MetaData()
        
        # Create documents table if it doesn't exist
        self.documents = Table(
            table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("file_path", String),
            Column("file_name", String),
            Column("file_type", String),
            Column("metadata", JSON),
            Column("created_at", DateTime),
            Column("updated_at", DateTime)
        )
        
        self.metadata.create_all(self.engine)
        
        # Set up file storage
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def store_documents(
        self,
        documents: List[Dict],
        copy_files: bool = True
    ) -> List[str]:
        """Store documents and their metadata.
        
        Args:
            documents: List of document dictionaries with metadata
            copy_files: Whether to copy the original files to storage
            
        Returns:
            List of document IDs
        """
        document_ids = []
        
        for doc in documents:
            try:
                # Generate document ID
                doc_id = self._generate_document_id(doc)
                
                # Copy file if requested
                if copy_files and "file_path" in doc:
                    new_path = self._copy_file_to_storage(
                        Path(doc["file_path"]),
                        doc_id
                    )
                    doc["file_path"] = str(new_path)
                
                # Store in database
                with self.engine.connect() as conn:
                    conn.execute(
                        self.documents.insert().values(
                            id=doc_id,
                            file_path=doc.get("file_path"),
                            file_name=doc.get("file_name"),
                            file_type=doc.get("file_type"),
                            metadata=doc.get("metadata", {}),
                            created_at=datetime.now(),
                            updated_at=datetime.now()
                        )
                    )
                    conn.commit()
                
                document_ids.append(doc_id)
                
            except Exception as e:
                logger.error(f"Error storing document: {str(e)}")
                continue
        
        return document_ids
    
    def retrieve_documents(
        self,
        document_ids: Optional[List[str]] = None,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve documents by ID or metadata filters.
        
        Args:
            document_ids: List of document IDs to retrieve
            metadata_filters: Metadata filters to apply
            
        Returns:
            List of document dictionaries
        """
        try:
            with self.engine.connect() as conn:
                if document_ids:
                    query = select(self.documents).where(
                        self.documents.c.id.in_(document_ids)
                    )
                elif metadata_filters:
                    # Build filter conditions
                    conditions = []
                    for key, value in metadata_filters.items():
                        conditions.append(
                            self.documents.c.metadata[key].astext == str(value)
                        )
                    query = select(self.documents).where(*conditions)
                else:
                    query = select(self.documents)
                
                result = conn.execute(query)
                documents = []
                
                for row in result:
                    doc = dict(row)
                    if "file_path" in doc and doc["file_path"]:
                        # Check if file exists
                        if Path(doc["file_path"]).exists():
                            doc["file_exists"] = True
                        else:
                            doc["file_exists"] = False
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def delete_documents(
        self,
        document_ids: List[str],
        delete_files: bool = True
    ) -> List[str]:
        """Delete documents and optionally their files.
        
        Args:
            document_ids: List of document IDs to delete
            delete_files: Whether to delete the stored files
            
        Returns:
            List of successfully deleted document IDs
        """
        deleted_ids = []
        
        try:
            with self.engine.connect() as conn:
                # Get file paths before deletion if needed
                if delete_files:
                    query = select(self.documents.c.file_path).where(
                        self.documents.c.id.in_(document_ids)
                    )
                    result = conn.execute(query)
                    file_paths = [row[0] for row in result if row[0]]
                
                # Delete from database
                stmt = delete(self.documents).where(
                    self.documents.c.id.in_(document_ids)
                )
                conn.execute(stmt)
                conn.commit()
                
                # Delete files if requested
                if delete_files:
                    for path in file_paths:
                        try:
                            if Path(path).exists():
                                Path(path).unlink()
                        except Exception as e:
                            logger.warning(f"Error deleting file {path}: {str(e)}")
                
                deleted_ids = document_ids
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
        
        return deleted_ids
    
    def _generate_document_id(self, document: Dict) -> str:
        """Generate a unique document ID based on content and metadata."""
        content = json.dumps(document, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _copy_file_to_storage(self, source_path: Path, doc_id: str) -> Path:
        """Copy a file to the storage directory with a unique name."""
        file_ext = source_path.suffix
        dest_path = self.storage_dir / f"{doc_id}{file_ext}"
        shutil.copy2(source_path, dest_path)
        return dest_path
