import re
import ast 
import pandas as pd

def parse_and_clean_tags(tags):
    if pd.isna(tags):
        return []
    if isinstance(tags, str):
        try:
            tags = ast.literal_eval(tags)
        except (ValueError, SyntaxError):
            tags = []
    
    if not isinstance(tags, list):
        tags = [str(tags)]
    
    cleaned = []
    for tag in tags:
        tag = str(tag).strip()
        tag = re.sub(r'["\'''""]', '', tag)
        tag = re.sub(r'[^a-zA-Z0-9\s]', ' ', tag)
        tag = re.sub(r'\s+', ' ', tag).strip()
        if tag:
            cleaned.append(tag)

    return cleaned

def build_search_text(title: str, description: str, tags: list = None, priority: str = None, department: str = None) -> str:
    tags_clean = parse_and_clean_tags(tags) if tags else []
    tags_part = ' '.join(tags_clean) if tags_clean else 'no tags'

    priority_part = f"{priority} priority" if priority else "unknown priority"
    department_part = f"{department} department" if department else "unknown department"
    
    text = f"{tags_part} {priority_part} {department_part}".strip()

    cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title.lower()).strip()[:100]
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
    if cleaned_title:
        text = f"{cleaned_title} {text}"
    
    cleaned_desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', description.lower()).strip()[:100]
    cleaned_desc = re.sub(r'\s+', ' ', cleaned_desc)

    if cleaned_desc:
        text = f"{text} {cleaned_desc}"

    return text
