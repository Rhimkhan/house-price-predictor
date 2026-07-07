import re

with open('src/data_preprocessing.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the indentation issue
content = content.replace('df_copy[col].fillna(median, inplace=True)', 'df_copy[col] = df_copy[col].fillna(median)')
content = content.replace('df_copy[col].fillna(mode, inplace=True)', 'df_copy[col] = df_copy[col].fillna(mode)')

with open('src/data_preprocessing.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed indentation errors!')
