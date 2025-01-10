import re
from typing import List, Dict, Union, Optional
import logging
from pathlib import Path
import json
import spacy
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self, log_path: Optional[str] = None):
        self.logger = self.setup_logging(log_path)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning('Downloading language model for the spaCy') 
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')

    
    def setup_logging(self, log_path: Optional[str]) -> logging.Logger:
        logger = logging.getLogger('preprocessor')
        logger.setLevel(logging.INFO)

        if log_path:
            handler = logging.FileHandler(log_path)
        else:
            handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    

    def clean_text(self, text:str) ->str:
        text = text.lower()
        # Removing special charcacters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = " ".join(text.split())
        return text
    
    def extract_features(self, structure: Dict) -> Dict:
        features = {
            'title_length': len(structure.get('title', '')),
            'abstract_length': len(structure.get('abstract', '')),
            'num_sections': len(structure.get('sections', [])),
            'avg_section_length': 0,
            'references_count': len(structure.get('references', [])),
            'technical_terms_count': 0,
            'methodology_score': 0,
            'results_score': 0
        }

        sections = structure.get('sections', [])
        if sections:
            total_length = sum(len(section.get('content', '')) for section in sections)
            features['avg_section_length'] = total_length / len(sections)

        
        all_text = ' '.join([
            structure.get('title', ''),
            structure.get('abstract', ''),
            *[section.get('content', '') for section in sections]
        ])

        features.update(self.analyze_content(all_text))
        return features
    

    def analyze_content(self, text: str) -> Dict:
        doc = self.nlp(text)

        analysis = {
            'technical_terms_count': 0,
            'methodology_score': 0,
            'results_score': 0
        }

        # Counting the technical terms
        technical_patterns = ['algorithm', 'methodology', 'analysis', 'experiment',
                            'data', 'results', 'conclusion', 'hypothesis']
        
        for token in doc:
            if token.lemma_.lower() in technical_patterns:
                analysis['technical_terms_count'] += 1

        # methodolgy score
        methodology_indicators = ['method', 'approach', 'procedure', 'experiment']
        analysis['methodology_score'] = sum(1 for word in methodology_indicators 
                                          if word in text.lower())
        
        # Result section score 
        results_indicators = ['result', 'finding', 'observation', 'analysis']
        analysis['results_score'] = sum(1 for word in results_indicators 
                                      if word in text.lower())
        
        return analysis
    

    def process_paper(self, input_path:str) ->Dict:
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                structure = json.load(f)
            
            features = self.extract_features(structure)
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            raise

    def batch_process(self, input_dir: str, output_dir: str) -> None:
        """Process all papers in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_features = []
        for file in input_path.glob('*.json'):
            try:
                features = self.process_paper(str(file))
                all_features.append(features)
                
                # Save individual features
                output_file = output_path / f"{file.stem}_features.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(features, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file}: {str(e)}")

        # Save combined features
        if all_features:
            combined_file = output_path / "all_features.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_features, f, indent=2)
