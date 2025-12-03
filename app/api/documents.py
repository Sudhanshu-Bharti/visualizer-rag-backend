from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
import uuid
import aiofiles
from datetime import datetime
from pathlib import Path
import logging

from app.models.schemas import DocumentUploadResponse, DocumentInfo, ProcessingStatus
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
from app.utils.file_security import (
    sanitize_filename,
    validate_file_extension,
    validate_file_content,
    validate_upload_path,
)

logger = logging.getLogger(__name__)

router = APIRouter()
document_processor = DocumentProcessor()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    # Validate content type
    if file.content_type not in [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Sanitize filename first
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    try:
        safe_filename = sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {str(e)}")

    # Validate file extension matches content type
    is_valid_ext, ext_error = validate_file_extension(safe_filename, file.content_type)
    if not is_valid_ext:
        raise HTTPException(status_code=400, detail=ext_error)

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Validate file size
    if file_size > settings.upload_max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,  # 413 Payload Too Large
            detail=f"File too large. Maximum size: {settings.upload_max_size_mb}MB",
        )

    # Validate file content matches declared type
    is_valid_content, content_error = validate_file_content(content, file.content_type)
    if not is_valid_content:
        raise HTTPException(
            status_code=400, detail=f"File content validation failed: {content_error}"
        )

    # Generate secure file path
    document_id = str(uuid.uuid4())
    filename = f"{document_id}_{safe_filename}"

    # Ensure uploads directory exists
    upload_dir = Path("uploads").resolve()
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / filename

    # Validate path is within uploads directory (prevent path traversal)
    is_valid_path, path_error = validate_upload_path(file_path, upload_dir)
    if not is_valid_path:
        logger.error(f"Path traversal attempt detected: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    file_path_str = str(file_path)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create document record in Neo4j immediately so it shows up in the list
    logger.info(f"Creating placeholder for document {document_id} ({safe_filename})")
    await document_processor.create_document_placeholder(
        document_id=document_id, filename=safe_filename, size=file_size
    )
    logger.info(f"Placeholder created successfully for {document_id}")

    # Add background task for document processing
    logger.info(f"Adding background task for document {document_id}")
    background_tasks.add_task(
        document_processor.process_document,
        document_id,
        file_path_str,
        safe_filename,
    )
    logger.info(f"Background task added successfully for {document_id}")

    return DocumentUploadResponse(
        id=document_id,
        filename=safe_filename,
        size=file_size,
        status="processing",
        created_at=datetime.utcnow(),
    )


@router.get("/", response_model=List[DocumentInfo])
async def list_documents():
    try:
        documents = await document_processor.get_all_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    try:
        document = await document_processor.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    try:
        success = await document_processor.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/status", response_model=ProcessingStatus)
async def get_processing_status(document_id: str):
    try:
        status = await document_processor.get_processing_status(document_id)
        if not status:
            raise HTTPException(status_code=404, detail="Document not found")
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
