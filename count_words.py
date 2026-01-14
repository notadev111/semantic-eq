import re

with open('docs/INTERIM_REPORT_CONCISE.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract main content (before References section)
refs_idx = content.find('## References')
if refs_idx > 0:
    main_content = content[:refs_idx]
else:
    main_content = content

# Remove markdown formatting
text = re.sub(r'!\[.*?\]\(.*?\)', '', main_content)  # Remove images
text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove links
text = re.sub(r'[#*_`]', '', text)  # Remove markdown symbols
text = re.sub(r'^\s*[-=]+\s*$', '', text, flags=re.MULTILINE)  # Remove dividers
text = re.sub(r'^\s*\|.*\|\s*$', '', text, flags=re.MULTILINE)  # Remove tables

# Count words
words = text.split()
print(f'Word count (excluding references): {len(words)}')
