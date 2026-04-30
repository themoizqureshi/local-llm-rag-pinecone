"""
Document loading helpers for the LlamaIndex pipeline.

LlamaIndex uses Document objects (not LangChain's).
SimpleDirectoryReader handles PDF, TXT, DOCX, and more automatically.
"""

import logging
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF into LlamaIndex Document objects.

    Each page becomes one Document with metadata (file_path, page_label).
    SimpleDirectoryReader uses pypdf under the hood for PDF parsing.
    """
    logger.info(f"Loading PDF: {file_path}")
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    logger.info(f"Loaded {len(documents)} pages")
    return documents


def load_directory(directory: str) -> List[Document]:
    """Load all supported documents from a directory."""
    logger.info(f"Loading documents from directory: {directory}")
    documents = SimpleDirectoryReader(directory).load_data()
    logger.info(f"Loaded {len(documents)} documents")
    return documents
