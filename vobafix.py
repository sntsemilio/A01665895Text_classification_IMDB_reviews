# Save this as clean_vocab.py and run: python clean_vocab.py
with open('vocab.txt', 'r', encoding='utf-8') as f:
    words = [line.rstrip('\n') for line in f]

# Remove ALL '' and '[UNK]' from everywhere
words = [w for w in words if w != '' and w != '[UNK]']

with open('vocab.txt', 'w', encoding='utf-8') as f:
    f.write('\n')          
    f.write('[UNK]\n')     
    for word in words:
        f.write(f'{word}\n')