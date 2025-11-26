import re

class TextCleaner:
    def __init__(self):
        self.abbrev_map = self._load_abbrev_map()
        self.noise_words = self._load_noise_words()
        self.misspellings = self._load_misspellings()

    def clean(self, title : str) -> str:
        """
        Full cleaning pipeline.
        Steps:
        1. lowercase
        2. remove duplicate
        3. expand abbreviations
        4. replace misspellings
        5. remove noise words
        6. remove levels (jr, sr, ii, iii, etc)
        7. normalize spaces
        """

        if not title:
            return ""
        
        text = title.lower().strip()
        text = self._remove_punctuation(text)
        text = self._expand_abbreviations(text)
        text = self._fix_misspellings(text)
        text = self._remove_noise(text)
        text = self._remove_levels(text)
        text = self._normalize_spaces(text)
        return text
    
    def _load_abbrev_map(self):
        """Mapping for abbreviation -> expanded form"""

        return {
           "ml": "machine learning",
           "dl": "deep learning",
           "ai": "artificial intelligence",
           "cv": "computer vision",
           "nlp": "natural language processing",
           "engg": "engineer",
           "engr": "engineer",
           "eng": "engineer",
           "dev": "developer",
           "prod": "product",
           "sci": "scientist",
           "sr": "senior",
           "jr": "junior",
           "bi": "business intelligence",
        }
    
    def _load_noise_words(self):
        """Words that don't matter for clustering."""
        return {
            "remote", "contract", "fulltime", "full-time",
            "hiring", "internship", "consultant"
        }
    
    def _load_misspellings(self):
        """Common misspellings -> correct versions."""
        return {
            "scntst": "scientist",
            "scintist": "scientist",
            "anlyst": "analyst",
            "analst": "analyst",
            "modeller": "modeler",
            "modler": "modeler",
            "architekt": "architect",
        }
    
    def _remove_punctuation(self,text):
        return re.sub(r"[^a-z0-9\s]", " ", text)
    
    def _expand_abbreviations(self, text):
        words = text.split()
        expanded = [self.abbrev_map.get(w, w) for w in words]
        return " ".join(expanded)
    
    def _fix_misspellings(self, text):
        words = text.split()
        fixed = [self.misspellings.get(w, w) for w in words]
        return " ".join(fixed)
    
    def _remove_noise(self, text):
        words = text.split()
        filtered = [w for w in words if w not in self.noise_words]
        return " ".join(filtered)
    
    def _remove_levels(self, text):
        # Removes "ii", "iii", "iv", "2", "lead", "principal"
        text = re.sub(r"\b(ii|iii|iv|2|3)\b", " ", text)
        text = re.sub(r"\b(lead|principal|manager)\b", " ", text)
        return text
    
    def _normalize_spaces(self, text):
        return re.sub(r"\s+", " ", text).strip()

 