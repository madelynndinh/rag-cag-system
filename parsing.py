import logging
from llmsherpa.readers import LayoutPDFReader
from typing import List
import os
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

# Use local server with new port
llmsherpa_api_url = "http://localhost:5011/api/parseDocument?renderFormat=all"

# Configure logging
logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str) -> List[Document]:
    """Parse a PDF file and convert it into a list of Document objects.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects containing the parsed content
    """
    try:
        logger.info(f"Parsing PDF file: {pdf_path}")
        
        # Initialize the PDF reader
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        
        # Read the PDF document
        doc = pdf_reader.read_pdf(pdf_path)
        documents = []
        current_section = None
        current_section_content = []
        
        # Process each block in the PDF
        for block in doc.json:
            block_text = ""
            block_metadata = {"type": block.get('tag', 'unknown')}
            
            # Handle different block types
            if block.get('tag') == 'header':
                # If we were building a section, save it
                if current_section and current_section_content:
                    section_text = "\n".join(current_section_content)
                    if section_text.strip():
                        documents.append(Document(
                            text=section_text,
                            metadata={
                                "type": "section",
                                "title": current_section,
                                "source": pdf_path,
                                "page_num": block.get('page_num', 0)
                            }
                        ))
                
                # Start new section
                block_text = ' '.join(block.get('sentences', []))
                current_section = block_text
                current_section_content = [block_text]
                block_metadata['level'] = block.get('level', 0)
            
            elif block.get('tag') == 'para':
                block_text = ' '.join(block.get('sentences', []))
                if current_section:
                    current_section_content.append(block_text)
            
            elif block.get('tag') == 'list_item':
                block_text = ' '.join(block.get('sentences', []))
                if current_section:
                    current_section_content.append(block_text)
                block_metadata['list_type'] = block.get('list_type', 'unknown')
            
            elif block.get('tag') == 'table':
                table_content = []
                for row in block.get('table_rows', []):
                    if row.get('type') == 'table_data_row':
                        cells = row.get('cells', [])
                        cell_values = [cell.get('cell_value', '') for cell in cells]
                        table_content.append(' | '.join(str(val) for val in cell_values))
                    elif row.get('type') == 'full_row':
                        table_content.append(row.get('cell_value', ''))
                block_text = '\n'.join(table_content)
                if current_section:
                    current_section_content.append(block_text)
                block_metadata['is_table'] = True
            
            # Create Document object if block has content and not part of a section
            if block_text.strip() and not current_section:
                block_metadata.update({
                    'source': pdf_path,
                    'page_num': block.get('page_num', 0),
                    'bbox': block.get('bbox', None)
                })
                
                documents.append(Document(
                    text=block_text,
                    metadata=block_metadata
                ))
        
        # Save the last section if exists
        if current_section and current_section_content:
            section_text = "\n".join(current_section_content)
            if section_text.strip():
                documents.append(Document(
                    text=section_text,
                    metadata={
                        "type": "section",
                        "title": current_section,
                        "source": pdf_path,
                        "page_num": block.get('page_num', 0)
                    }
                ))
        
        logger.info(f"Successfully parsed PDF into {len(documents)} document blocks")
        return documents
        
    except Exception as e:
        logger.error(f"Error parsing PDF file {pdf_path}: {str(e)}")
        raise

def print_section_contents(pdf_path: str):
    """Print the content of each section after parsing a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    """
    try:
        documents = parse_pdf(pdf_path)
        
        print(f"\nParsed {len(documents)} document blocks from {pdf_path}")
        print("\n" + "="*80)
        print(f"SECTIONS IN {os.path.basename(pdf_path)}")
        print("="*80)
        
        for i, doc in enumerate(documents):
            print(f"\nSECTION {i+1}:")
            print(f"Title: {doc.metadata.get('title', 'No title')}")
            print(f"Type: {doc.metadata.get('type', 'Unknown type')}")
            print(f"Page: {doc.metadata.get('page_num', 'Unknown page')}")
            print("-"*40)
            print(doc.text[:500] + ("..." if len(doc.text) > 500 else ""))
            print("-"*40)
            
    except Exception as e:
        print(f"Error printing section contents for {pdf_path}: {str(e)}")

# Demonstration code for parsing CR2024-009.pdf
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(levelname)s:%(name)s:%(message)s')
    
    # Specify the path to the PDF file
    pdf_file = "pdf-test/cr2024-009.pdf"
    
    # Print content in each section after parsing
    print_section_contents(pdf_file)
