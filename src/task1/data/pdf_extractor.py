import os
import fitz
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path 
import re

class PdfExtractor:
    def __init__(self, log_path: Optional[str] = None):
        # Setting up a logger with a optional log file path
        self.logger = self.setup_logger(log_path)

    def setup_logging(self, log_path: Optional[str]) -> logging.logger:
        # Creating a logger named pdfExtractor
        logger = logging.getLogger('pdfExtractor')
        logger.setLevel(logging.INFO)  # Setting the log level to INFO

        # Creating a file handler or a stream handler console based on the log_path
        if log_path:
            handler = logging.FileHandler(log_path) # Redirects the logs to a file
        else:
            handler = logging.StreamHandler() # Stream handler to print the logs on console

        # Creating a message format for the logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger 
    
    # Main Extraction Function 
    def extract_structure(self, pdf_path: str) -> Dict:
        try:
            # Opening the pdf file
            doc = fitz.open(pdf_path)

            #Inializing the dictionary to store the stucture of the pdf
            structure = {
                'tile': '',
                'abstract': '',
                'sections': [],
                'references': [],
                'metadata': self.extract_metadata(doc)
            }

            # Collecting all the text blocks from each page 
            text_blocks = []
            for page in doc:
                # Getting the page content as a dictionary
                blocks = page.get_text("dict"["blocks"])
                # Process and adding blocks to the colleection
                text_blocks.extend(self.process_blocks(blocks))

            # Orgainzing the blocks into a structured content
            structure.update(self.orgainze_content(text_blocks))

            doc.close()
            self.logger.info(f"Successfully xtracted structure from {pdf_path}")
            return structure
        
        except Exception as e:
            self.logger.error(f"Error extracting the structure from {pdf_path}: {str(e)}")
            raise

    # Metadata Extraction Function:
    def extract_metadata(self, doc: fitz.Document) -> Dict:
        # Extrcting the basic metadata from the pdf
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'creation_date': doc.metadata.get('creation_date', ''),
            'modification_date': doc.metadata.get('modification_date', ''),
            'page_count': len(doc) #Number of pages
        }
        return metadata