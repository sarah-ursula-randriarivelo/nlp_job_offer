"""
Skill Extraction Module
=======================
Extract competencies and qualifications from job offers.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Set, Optional
import spacy
from collections import Counter

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("⚠️  Modèle spaCy 'fr_core_news_sm' non trouvé.")
    nlp = None


# Common skills in Madagascar job market
TECHNICAL_SKILLS = {
    # Programming & Development
    'python', 'java', 'javascript', 'typescript', 'php', 'c#', 'c++', 'ruby', 'go', 'rust',
    'html', 'css', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'laravel',
    
    # Data & Analytics
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'power bi', 'tableau', 'excel', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github', 'ci/cd',
    
    # Business & Finance
    'sap', 'erp', 'crm', 'comptabilite', 'audit', 'fiscalite', 'tresorerie', 'budget',
    
    # Marketing & Communication
    'seo', 'sem', 'google analytics', 'facebook ads', 'linkedin', 'mailchimp', 'wordpress',
    
    # Design & Creative
    'photoshop', 'illustrator', 'indesign', 'figma', 'canva', 'premiere pro'
}

SOFT_SKILLS = {
    'communication', 'leadership', 'travail equipe', 'autonomie', 'rigueur', 'organisation',
    'gestion temps', 'adaptabilite', 'creativite', 'analyse', 'problem solving', 'negociation',
    'relationnel', 'dynamisme', 'motivation', 'proactivite', 'esprit initiative'
}

QUALIFICATIONS = {
    # Diplomas
    'licence', 'master', 'doctorat', 'bac', 'bts', 'dut', 'ingenieur', 'mba',
    
    # Certifications
    'certification', 'certifie', 'agile', 'scrum', 'pmp', 'itil', 'cisco', 'aws certified',
    
    # Experience levels
    'junior', 'senior', 'expert', 'debutant', 'confirme', 'stagiaire', 'alternance',
    
    # Languages
    'francais', 'anglais', 'malgache', 'bilingue', 'trilingue'
}


class SkillExtractor:
    """Extract skills and qualifications from job descriptions."""
    
    def __init__(self, 
                 technical_skills: Optional[Set[str]] = None,
                 soft_skills: Optional[Set[str]] = None,
                 qualifications: Optional[Set[str]] = None):
        """
        Initialize skill extractor.
        
        Args:
            technical_skills: Set of technical skills to look for
            soft_skills: Set of soft skills to look for
            qualifications: Set of qualifications to look for
        """
        self.technical_skills = technical_skills or TECHNICAL_SKILLS
        self.soft_skills = soft_skills or SOFT_SKILLS
        self.qualifications = qualifications or QUALIFICATIONS
    
    def extract_skills_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from a single text.
        
        Args:
            text: Job description text
            
        Returns:
            Dictionary with 'technical', 'soft', and 'qualifications' keys
        """
        if not isinstance(text, str):
            return {'technical': [], 'soft': [], 'qualifications': []}
        
        text_lower = text.lower()
        
        # Find skills
        found_technical = [skill for skill in self.technical_skills if skill in text_lower]
        found_soft = [skill for skill in self.soft_skills if skill in text_lower]
        found_qualifications = [qual for qual in self.qualifications if qual in text_lower]
        
        return {
            'technical': found_technical,
            'soft': found_soft,
            'qualifications': found_qualifications
        }
    
    def extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text.
        
        Args:
            text: Job description text
            
        Returns:
            Number of years of experience or None
        """
        if not isinstance(text, str):
            return None
        
        # Patterns for experience
        patterns = [
            r'(\d+)\s*(?:ans?|années?)\s*(?:d\'?)?(?:experience|expérience)',
            r'experience\s*(?:de\s*)?(\d+)\s*ans?',
            r'(\d+)\+\s*(?:ans?|années?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def extract_education_level(self, text: str) -> Optional[str]:
        """
        Extract education level from text.
        
        Args:
            text: Job description text
            
        Returns:
            Education level or None
        """
        if not isinstance(text, str):
            return None
        
        text_lower = text.lower()
        
        education_hierarchy = [
            ('doctorat', 'Doctorat'),
            ('phd', 'Doctorat'),
            ('master', 'Master'),
            ('mba', 'MBA'),
            ('ingenieur', 'Ingénieur'),
            ('licence', 'Licence'),
            ('bachelor', 'Licence'),
            ('bts', 'BTS'),
            ('dut', 'DUT'),
            ('bac', 'Baccalauréat')
        ]
        
        for keyword, level in education_hierarchy:
            if keyword in text_lower:
                return level
        
        return None
    
    def extract_languages(self, text: str) -> List[str]:
        """
        Extract language requirements from text.
        
        Args:
            text: Job description text
            
        Returns:
            List of languages
        """
        if not isinstance(text, str):
            return []
        
        text_lower = text.lower()
        languages = []
        
        language_keywords = {
            'francais': 'Français',
            'français': 'Français',
            'anglais': 'Anglais',
            'english': 'Anglais',
            'malgache': 'Malgache',
            'malagasy': 'Malgache'
        }
        
        for keyword, lang in language_keywords.items():
            if keyword in text_lower and lang not in languages:
                languages.append(lang)
        
        return languages
    
    def extract_all_from_dataframe(self, 
                                  df: pd.DataFrame,
                                  text_column: str = 'mission_cleaned',
                                  title_column: str = 'title_cleaned') -> pd.DataFrame:
        """
        Extract all skills and qualifications from DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing job description
            title_column: Column containing job title
            
        Returns:
            DataFrame with extracted features
        """
        print(f"🔍 Extracting skills from {len(df)} job offers...")
        
        results = []
        
        for idx, row in df.iterrows():
            # Combine title and description for extraction
            combined_text = f"{row.get(title_column, '')} {row.get(text_column, '')}"
            
            # Extract skills
            skills = self.extract_skills_from_text(combined_text)
            
            # Extract other info
            years_exp = self.extract_years_experience(combined_text)
            education = self.extract_education_level(combined_text)
            languages = self.extract_languages(combined_text)
            
            result = {
                'technical_skills': ', '.join(skills['technical']) if skills['technical'] else '',
                'soft_skills': ', '.join(skills['soft']) if skills['soft'] else '',
                'qualifications': ', '.join(skills['qualifications']) if skills['qualifications'] else '',
                'technical_skills_count': len(skills['technical']),
                'soft_skills_count': len(skills['soft']),
                'years_experience': years_exp,
                'education_level': education,
                'languages': ', '.join(languages) if languages else ''
            }
            
            results.append(result)
        
        # Create DataFrame and concatenate with original
        skills_df = pd.DataFrame(results)
        result_df = pd.concat([df.reset_index(drop=True), skills_df], axis=1)
        
        print("✅ Skill extraction complete!")
        print(f"   Technical skills found: {skills_df['technical_skills_count'].sum()}")
        print(f"   Soft skills found: {skills_df['soft_skills_count'].sum()}")
        
        return result_df
    
    def get_skill_frequency(self, df: pd.DataFrame, skill_type: str = 'technical') -> pd.DataFrame:
        """
        Get frequency of skills across all job offers.
        
        Args:
            df: DataFrame with extracted skills
            skill_type: 'technical', 'soft', or 'qualifications'
            
        Returns:
            DataFrame with skill frequencies
        """
        column = f'{skill_type}_skills'
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Run extract_all_from_dataframe first.")
        
        # Collect all skills
        all_skills = []
        for skills_str in df[column].dropna():
            if skills_str:
                all_skills.extend([s.strip() for s in skills_str.split(',')])
        
        # Count frequency
        skill_counts = Counter(all_skills)
        
        freq_df = pd.DataFrame(skill_counts.most_common(), columns=['skill', 'frequency'])
        freq_df['percentage'] = freq_df['frequency'] / len(df) * 100
        
        return freq_df
    
    def get_cluster_skills(self, 
                          df: pd.DataFrame,
                          cluster_id: int,
                          skill_type: str = 'technical',
                          top_n: int = 10) -> pd.DataFrame:
        """
        Get top skills for a specific cluster.
        
        Args:
            df: DataFrame with clusters and extracted skills
            cluster_id: Cluster ID
            skill_type: 'technical', 'soft', or 'qualifications'
            top_n: Number of top skills to return
            
        Returns:
            DataFrame with top skills for cluster
        """
        cluster_df = df[df['cluster'] == cluster_id]
        
        column = f'{skill_type}_skills'
        all_skills = []
        
        for skills_str in cluster_df[column].dropna():
            if skills_str:
                all_skills.extend([s.strip() for s in skills_str.split(',')])
        
        skill_counts = Counter(all_skills).most_common(top_n)
        
        return pd.DataFrame(skill_counts, columns=['skill', 'count'])


def extract_skills(df: pd.DataFrame,
                  text_column: str = 'mission_cleaned',
                  title_column: str = 'title_cleaned') -> pd.DataFrame:
    """
    Convenience function to extract skills from DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Column containing job description
        title_column: Column containing job title
        
    Returns:
        DataFrame with extracted skills
    """
    extractor = SkillExtractor()
    return extractor.extract_all_from_dataframe(df, text_column, title_column)
