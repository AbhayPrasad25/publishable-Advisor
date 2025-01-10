import os
import fitz
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path 
import re

class PdfExtractor:
    def __init__(self, log_path: Optional[str] = None):
        # Setting up a logger with a optional log file path
        self.logger = self.setup_logger(log_path)  # Rename to match method definition

    def setup_logger(self, log_path: Optional[str]) -> logging.Logger:
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
                'title': '',
                'abstract': '',
                'sections': [],
                'references': [],
                'metadata': self.extract_metadata(doc)
            }

            # Collecting all the text blocks from each page 
            text_blocks = []
            for page in doc:
                # Getting the page content as a dictionary
                blocks = page.get_text("dict")["blocks"]
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
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'page_count': len(doc) #Number of pages
        }
        return metadata
    
    # Processing the text blocks
    def process_blocks(self, blocks: List) -> List[Dict]:
        procesed_blocks = []
        for block in blocks:
            if block.get('type') == 0: # Chcecking wether it is a text block
                for line in block.get('lines', []):
                    text_segments = []
                    for span in line.get('spans', []):

                        # Extracting the text and its formatting
                        text_segments.append({
                            'text': span.get('text', ''),
                            'font': span.get('font', ''),
                            'size': span.get('size', 0),
                            'flags': span.get('flags', 0)
                        })
                    
                    procesed_blocks.append(text_segments)
        return procesed_blocks
    

    
    # Organizing the content into a structured format
    def orgainze_content(self, text_blocks: List) -> Dict:
        structure = {
            'sections' : [],
            'current_section': None
        }

        for block in text_blocks:
            # Checking if the block is a heading
            if self.is_heading(block):
                # If current section add it to the section list
                if structure['current_section']:
                    structure['sections'].append(structure['current_section'])

                # New Section
                structure['current_section'] = {
                    'heading': self.get_text(block),
                    'content': [],
                    'subsections': []
                }

            elif self.is_subheading(block):
                if structure['current_section']:
                    structure['current_section']['subsections'].append(self.get_text(block))

            elif structure['current_section']:
                structure['current_section']['content'].append(self.get_text(block))

        # Adding the last section
        if structure['current_section']:
            structure['sections'].append(structure['current_section'])
        
        return structure 
    
    # Checking the block is a heading or not 
    def is_heading(self, block: List[Dict]) -> bool:
        # Using the text formatting to check if the block is a heading

        if not block or not isinstance(block, list):
            return False
        
        # Calcualting the average font size and checking for bold formatting
        avgerage_font_size = sum(span['size'] for span in block) / len(block)
        is_bold = any(span.get('flags', 0) & 2 for span in block)

        return avgerage_font_size > 12 or is_bold
    
    # Checking the block is a subheading or not
    def is_subheading(self, block: List[Dict]) -> bool:
        text = self.get_text(block)

        subsection_pattern = r'^\d+(\.\d+)*[A-Za-z]*\.$'

        return bool(re.match(subsection_pattern, text.strip()))
    
    # Extracting the text from the block
    def get_text(self, block: List[Dict]) -> str:
        return ' '.join(span['text'] for span in block)
    
    # Batch Processing of the pdf files
    def batch_process(self, input_directory: str, output_directory: str) -> None:
        # Creating the output directory if it does not exist
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # Processing each pdf file in the input directory
        for filename in os.listdir(input_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_directory, filename)
                output_path = os.path.join(output_directory, f"{filename[:-4]}.json")

                try:
                    structure = self.extract_structure(pdf_path)
                    self.save_structure(structure, output_path)
                except Exception as e:
                    self.logger.error(f"Failed to process {filename}: {str(e)}")

    # Saving the structure to a json file
    def save_structure(self, structure: Dict, output_path: str) -> None:
        
        import json
        with open(output_path, 'w', encoding = 'utf-8') as f:
            json.dump(structure, f, ensure_ascii= False, indent =  2)