from typing import Dict, List, Union, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging
from pathlib import Path
import json
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter

class FeatureExtractor:
    def __init__(self, 
                 use_tfidf: bool = True,
                 use_doc2vec: bool = True,
                 doc2vec_vector_size: int = 100,
                 log_path: Optional[str] = None):
        """
        Initialize the feature extractor with desired feature types
        
        Args:
            use_tfidf: Whether to use TF-IDF features
            use_doc2vec: Whether to use Doc2Vec embeddings
            doc2vec_vector_size: Size of Doc2Vec vectors
            log_path: Path to log file (optional)
        """
        self.use_tfidf = use_tfidf
        self.use_doc2vec = use_doc2vec
        self.doc2vec_vector_size = doc2vec_vector_size
        
        # Initialize components
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=100)  # For dimensionality reduction of TF-IDF
        self.doc2vec = None
        
        # Load spaCy model for text processing
        self.nlp = spacy.load('en_core_web_sm')
        
        # Setup logging
        self.logger = self._setup_logging(log_path)

    def _setup_logging(self, log_path: Optional[str]) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('FeatureExtractor')
        logger.setLevel(logging.INFO)
        
        if log_path:
            handler = logging.FileHandler(log_path)
        else:
            handler = logging.StreamHandler()
            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def extract_structural_features(self, paper_structure: Dict) -> Dict[str, float]:
        """Extract structural features from paper content"""
        features = {}
        
        # Basic length features
        features['title_length'] = len(paper_structure.get('title', '').split())
        features['abstract_length'] = len(paper_structure.get('abstract', '').split())
        
        # Section analysis
        sections = paper_structure.get('sections', [])
        features['num_sections'] = len(sections)
        
        section_lengths = [len(section.get('content', '').split()) for section in sections]
        features['avg_section_length'] = np.mean(section_lengths) if section_lengths else 0
        features['max_section_length'] = max(section_lengths) if section_lengths else 0
        features['min_section_length'] = min(section_lengths) if section_lengths else 0
        
        # Reference analysis
        references = paper_structure.get('references', [])
        features['num_references'] = len(references)
        
        return features

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text content"""
        doc = self.nlp(text)
        
        features = {}
        
        # Sentence complexity
        sentences = list(doc.sents)
        sentence_lengths = [len(sent) for sent in sentences]
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        features['max_sentence_length'] = max(sentence_lengths) if sentence_lengths else 0
        
        # Part of speech distributions
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)
        
        for pos, count in pos_counts.items():
            features[f'pos_ratio_{pos.lower()}'] = count / total_tokens if total_tokens > 0 else 0
        
        # Named entity analysis
        ent_counts = Counter([ent.label_ for ent in doc.ents])
        features['num_entities'] = len(doc.ents)
        
        for ent_type, count in ent_counts.items():
            features[f'ent_count_{ent_type.lower()}'] = count
        
        return features

    def train_doc2vec(self, documents: List[str]) -> None:
        """Train Doc2Vec model on the corpus"""
        if not self.use_doc2vec:
            return
            
        tagged_docs = [TaggedDocument(doc.split(), [i]) 
                      for i, doc in enumerate(documents)]
        
        self.doc2vec = Doc2Vec(vector_size=self.doc2vec_vector_size,
                              min_count=2,
                              epochs=40)
        
        self.doc2vec.build_vocab(tagged_docs)
        self.doc2vec.train(tagged_docs,
                          total_examples=self.doc2vec.corpus_count,
                          epochs=self.doc2vec.epochs)

    def get_document_embedding(self, text: str) -> np.ndarray:
        """Get document embedding using trained Doc2Vec model"""
        if not self.use_doc2vec or self.doc2vec is None:
            return np.zeros(self.doc2vec_vector_size)
        
        return self.doc2vec.infer_vector(text.split())

    def extract_text_features(self, corpus: List[str]) -> Dict[str, np.ndarray]:
        """Extract text-based features (TF-IDF and Doc2Vec)"""
        features = {}
        
        if self.use_tfidf:
            # TF-IDF features with dimensionality reduction
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            features['tfidf'] = self.svd.fit_transform(tfidf_matrix)
        
        if self.use_doc2vec:
            # Train Doc2Vec if not already trained
            if self.doc2vec is None:
                self.train_doc2vec(corpus)
            
            # Get Doc2Vec embeddings for each document
            features['doc2vec'] = np.vstack([
                self.get_document_embedding(doc) for doc in corpus
            ])
        
        return features

    def process_paper(self, paper_structure: Dict) -> Dict:
        """Process a single paper and extract all features"""
        try:
            # Combine all text content
            text_content = ' '.join([
                paper_structure.get('title', ''),
                paper_structure.get('abstract', ''),
                *[section.get('content', '') 
                  for section in paper_structure.get('sections', [])]
            ])
            
            # Extract different types of features
            structural_features = self.extract_structural_features(paper_structure)
            linguistic_features = self.extract_linguistic_features(text_content)
            
            # Combine all features
            features = {
                **structural_features,
                **linguistic_features
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing paper: {str(e)}")
            raise

    def batch_process(self, 
                     input_dir: str, 
                     output_dir: str,
                     extract_embeddings: bool = True) -> None:
        """Process all papers in directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_features = []
        all_texts = []

        # First pass: collect all text and extract basic features
        for file in input_path.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    paper_structure = json.load(f)
                
                # Extract features
                features = self.process_paper(paper_structure)
                all_features.append(features)
                
                # Collect text for embedding features
                if extract_embeddings:
                    text_content = ' '.join([
                        paper_structure.get('title', ''),
                        paper_structure.get('abstract', ''),
                        *[section.get('content', '') 
                          for section in paper_structure.get('sections', [])]
                    ])
                    all_texts.append(text_content)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file}: {str(e)}")

        # Extract text-based features if requested
        if extract_embeddings and all_texts:
            text_features = self.extract_text_features(all_texts)
            
            # Add embeddings to features
            for i, features in enumerate(all_features):
                if 'tfidf' in text_features:
                    features['tfidf_embedding'] = text_features['tfidf'][i].tolist()
                if 'doc2vec' in text_features:
                    features['doc2vec_embedding'] = text_features['doc2vec'][i].tolist()

        # Save individual features
        for i, features in enumerate(all_features):
            output_file = output_path / f"paper_{i+1}_features.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=2)

        # Save combined features
        if all_features:
            combined_file = output_path / "all_features.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_features, f, indent=2)

        self.logger.info(f"Processed {len(all_features)} papers successfully")