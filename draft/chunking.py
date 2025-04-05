from typing import List, Dict, Any, Optional, Generator,  Tuple
from llama_index.core import Document
import config
from pathlib import Path
import re
import logging
import gc
import uuid
import logging
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import Document
from enum import Enum


logger = logging.getLogger(__name__)
    
class DocumentDTO:
    """
    DocumentDTO_for_parent_docs is a data transfer object that is used to store the document for the parent documents.
    
    Args:
        Collection_name (str): Collection name of the document.
        Source (str): Source of the document (Main Source: Can be main URL, PDF/word/Excel/CSV file path, etc.).
        Page_content (str): Page content of the document.
        Parent_uuid (str): Parent UUID of the document.
        Sub_url (str, optional): Sub-URL of the document. Defaults to None. Only available for Websites.
        Starting_page_of_parent (int, optional): Starting page of the parent document. Defaults to None. Only available for PDF/Word files.
        Ending_page_of_parent (int, optional): Ending page of the parent document. Defaults to None. Only available for PDF/Word files.
        Row_no (int, optional): Row number of the document. Defaults to None. Only available for Excel/CSV files.
        Sheet_name (str, optional): Sheet name of the document. Defaults to None. Only available for Excel files.

    Returns:
        DocumentDTO_for_parent_docs: DocumentDTO_for_parent_docs object
    """    
    def __init__(self, collection_name: str, source: str, page_content: str, parent_uuid: str, sub_url: str = None, starting_page_of_parent: int = None, ending_page_of_parent: int = None, row_no: int = None, sheet_name: str = None) -> None:
        self.collection_name = collection_name
        self.source = source
        self.sub_url = sub_url
        self.page_content = page_content
        self.parent_uuid = parent_uuid
        self.starting_page_of_parent = starting_page_of_parent
        self.ending_page_of_parent = ending_page_of_parent
        self.row_no = row_no
        self.sheet_name = sheet_name
        
    def to_dict(self, return_non_empty_values: bool = False, exclude: List[str] = []):
        """
        Returns the dictionary object of the DocumentDTO_for_parent_docs object.

        Args:
            return_non_empty_values (bool, optional): If True, returns only non-empty values. Defaults to False.
        """
        return_dict = {}
        if return_non_empty_values:
            for key, value in self.__dict__.items():
                if value and key not in exclude:
                    return_dict[key] = value
        else:
            return_dict = self.__dict__
            return_dict = {key: value for key, value in return_dict.items() if key not in exclude}
        return return_dict

    def from_dict(self, data: dict):
        """
        Update the DocumentDTO_for_parent_docs object from a dictionary.

        Args:
            data (dict): Dictionary containing the data to update the object.
        """
        for key, value in data.items():
            setattr(self, key, value)
        return self
    



class ChunkingInfo(Enum):
    """
    Hyperparameters required for working with Chunking.
    
    Attributes:
        CHUNK_SIZE (int): Default to 500 for chunking, paragraphs chunk size.
        CHUNK_OVERLAP (int): Default to 50 for chunking.
    """
    CHUNK_SIZE = 1000 # Default to 1000 for chunking.
    CHUNK_OVERLAP = 200 # Default to 200 for chunking.
    CHUNK_BASED_ON_THESE_CHARACTERS = ["\n"]


class ChunkingService:
    """
    Chunking Service.
    """
    def generate_parent_document_for_given_documents(self, documents: List[Document], APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT: int = 2000):
        # Initialize parser
        markdown_parser = MarkdownNodeParser(
            heading_splitter_levels=[1, 2, 3, 4],
            include_metadata=True,
        )
        
        parent_uuid = str(uuid.uuid4())
        parent_docs = []
        all_nodes = []
        
        # Process documents
        for doc in documents:
            # Parse into nodes with hierarchy
            nodes = markdown_parser.get_nodes_from_documents([doc])
            
            # Add metadata to nodes
            for node in nodes:
                # Add parent_uuid and source metadata
                node.metadata.update({
                    "parent_uuid": parent_uuid,
                    "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "level": len(node.metadata.get("heading_hierarchy", [])),  # Set level based on heading hierarchy
                })
                all_nodes.append(node)
                
                # Create parent document if this is a top-level node
                if node.metadata.get("level", 0) == 0:
                    parent_docs.append(node)
                
        return parent_docs, all_nodes
        
    def chunk_website_documents(self, documents: List[Document], APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT: int = 2000) -> Tuple[List[Document], List[Document]]:
        """
        Chunk the website documents.

        Args:
            documents (List[Document]): The list of documents to chunk.
                - These are URL-wise documents, where each document is a URL content.
            APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT (int): The approximate number of tokens allowed per parent document. Defaults to 2000.

        Returns:
            Tuple[List[Document], List[Document]]: The parent documents and chunked documents.
        """
        try:
            logger.info("Chunking website documents | No of documents: %s", len(documents))
            
            # Initialisation
            parent_docs, overall_chunked_docs = [], []
            
            # Define Markdown headers to split on
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4")
            ]
            markdown_splitter = MarkdownNodeParser(headers_to_split_on=headers_to_split_on, 
                                                 strip_headers=False)
        
            # Calling the generate_parent_document_for_given_documents method for each document
            for document in documents:
                parent_docs_for_document, overall_chunked_docs_for_document = self.generate_parent_document_for_given_documents(
                    documents=[document], 
                    APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT=APPROX_ALLOWED_TOKENS_PER_PARENT_DOCUMENT
                )
                parent_docs.extend(parent_docs_for_document)
                overall_chunked_docs.extend(overall_chunked_docs_for_document)
            
            logger.info("[Chunking] Level 1 Chunking (Markdown Header Text Splitter) completed | No of parent documents: %s | No of chunked documents: %s", 
                       len(parent_docs), len(overall_chunked_docs))
            
            # Second level chunking: SentenceSplitter
            sentence_splitter = SentenceSplitter(chunk_size=ChunkingInfo.CHUNK_SIZE.value, 
                                               chunk_overlap=ChunkingInfo.CHUNK_OVERLAP.value)
            
            # Convert nodes to documents while preserving metadata
            final_chunks = []
            for doc in overall_chunked_docs:
                chunks = sentence_splitter.split_text(doc.text)
                for chunk in chunks:
                    # Create new document with preserved metadata
                    chunk_doc = Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "is_sentence_chunk": True
                        }
                    )
                    final_chunks.append(chunk_doc)
                    
            logger.info("[Chunking] Level 2 Chunking (Sentence Splitter) completed | No of Chunked Documents: %s", len(final_chunks))
            
            return parent_docs, final_chunks
        except Exception as e:
            logger.error("Error in chunking website documents | Error: %s", e)
            return [], []
        


        """Calculates the no of tokens required to tokenize a given text."""

from typing import List, Union

import tiktoken

# Document types that can be used for tokenization
from llama_index.core import Document
class TokenizerInfo(Enum):
    """
    Hyperparameters required for working with Tokenizer.
    
    Attributes:
        NO_OF_THREADS_FOR_TOKENIZATION (int): Number of threads for tokenization.
        ENCODING_MODEL_NAME (str): Encoding Model Name for Tokenization.
    """
    NO_OF_THREADS_FOR_TOKENIZATION = 4 # Default to 4 threads for tokenization.
    ENCODING_MODEL_NAME = "o200k_base" # For gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large models.

class TokenizerService:
    """
    TokenizerService class to calculate the no of tokens required to tokenize a given text.
    """
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding(encoding_name=TokenizerInfo.ENCODING_MODEL_NAME.value)

    def encode_text_to_tokens(self, documents: List[Union[str, Document, DocumentDTO]]) -> List[List[int]]:
        """
        Encodes the text to tokens.
        
        Args:
            documents (List[Union[str, DocumentDto]]): List of documents to be encoded.
            
        Returns:
            List[List[int]]: List of tokens.
        """
        # Convert the documents to strings
        documents = self.__convert_documents_to_string(documents)
        
        # Encode the text to tokens
        tokens = self.tokenizer.encode_batch(text = documents, num_threads=TokenizerInfo.NO_OF_THREADS_FOR_TOKENIZATION.value)
        
        # Return the tokens
        return tokens
    
    def get_tokens_count_for_each_document(self, documents: List[Union[str,  DocumentDTO]]) -> List[int]:
        """
        Return the no of tokens required to tokenize each document.
        
        Args:
            documents (List[Union[str, Document, DocumentDto, DocumentDTOForCache]]): List of documents to be tokenized.
            
        Returns:
            List[int]: List of no of tokens required to tokenize each document.
        """
        # Get the tokens
        tokens = self.encode_text_to_tokens(documents)
        
        # Get the no of tokens for each document
        no_of_tokens = [len(token) for token in tokens]
        
        # Return the no of tokens for each document
        return no_of_tokens

    def get_tokens_count(self, documents: List[Union[str,  DocumentDTO]]) -> int:
        """
        Return the no of tokens required to tokenize entire documents (Summed up result for all documents).
        
        Args:
            documents (List[Union[str,  DocumentDto]]): List of documents to be tokenized.
            
        Returns:
            int: No of tokens required to tokenize the entire documents.
        """
        # Sum up the tokens
        no_of_tokens = sum(self.get_tokens_count_for_each_document(documents))
        
        # Return the no of tokens
        return no_of_tokens
    
    
    def __convert_documents_to_string(self, documents: List[Union[str, DocumentDTO]]) -> List[str]:
        """
        Converts the list of Documents to list of strings.
        
        Args:
            documents (List[Union[str, Document, DocumentDto, ]): List of Documents to be converted to strings.
            
        Returns:
            List[str]: List of strings.
        """
        if not documents:
            return []
        
        if isinstance(documents[0], str):
            return documents
        elif isinstance(documents[0], Document):
            return [document.text for document in documents]
        elif isinstance(documents[0], DocumentDTO):
            return [document.document for document in documents]
     