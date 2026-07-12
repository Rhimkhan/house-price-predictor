content = open('src/api/index.py', 'rb').read().decode('utf-8-sig')
print('Has INR:', '* 84' in content)
print('Has rupee:', 'rupee' in content.lower() or '\u20b9' in content)
print('Dollar sign:', '\"\$\"' in content or '\"\\$\"' in content)
