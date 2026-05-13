import re
import pandas as pd

if __name__ == '__main__':
    with open('PMC2_content.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    test_samples = []
    train_samples = []
    
    # Test data extraction (samples 1 to 18)
    # They appear after line 70 "Amostra"
    # The columns are: Amostra, x1, x2, x3, x4, d1, d2, d3, y1, y2, y3
    # Wait, the table header has:
    # 70: Amostra
    # 71: x1
    # 72: x2
    # 73: x3
    # 74: x4
    # 75: d1
    # 76: d2
    # 77: d3
    # 78: y1
    # 79: y2
    # 80: y3
    
    idx = 0
    while idx < len(lines):
        if lines[idx] == 'y3':
            idx += 1
            break
        idx += 1
        
    for i in range(1, 19):
        while idx < len(lines):
            if lines[idx] == str(i):
                try:
                    x1 = float(lines[idx+1])
                    x2 = float(lines[idx+2])
                    x3 = float(lines[idx+3])
                    x4 = float(lines[idx+4])
                    d1 = float(lines[idx+5])
                    d2 = float(lines[idx+6])
                    d3 = float(lines[idx+7])
                    test_samples.append((i, x1, x2, x3, x4, d1, d2, d3))
                    idx += 8
                    break
                except:
                    idx += 1
            else:
                idx += 1
                
    test_df = pd.DataFrame(test_samples, columns=['id', 'x1', 'x2', 'x3', 'x4', 'd1', 'd2', 'd3'])
    test_df.to_csv('test.csv', index=False)
    
    # Train data extraction
    # They appear after "ANEXO"
    anexo_idx = 0
    for i, line in enumerate(lines):
        if line == 'ANEXO':
            anexo_idx = i
            break
            
    idx = anexo_idx
    while idx < len(lines):
        if re.match(r'^\d+$', lines[idx]):
            sample_id = int(lines[idx])
            # Assuming max 150 samples in train set. Actually it says 148 ensaios?
            # 18 in test, so 130 in train. Total = 148. Correct.
            if 1 <= sample_id <= 130:
                try:
                    x1 = float(lines[idx+1])
                    x2 = float(lines[idx+2])
                    x3 = float(lines[idx+3])
                    x4 = float(lines[idx+4])
                    d1 = float(lines[idx+5])
                    d2 = float(lines[idx+6])
                    d3 = float(lines[idx+7])
                    train_samples.append((sample_id, x1, x2, x3, x4, d1, d2, d3))
                    idx += 7 # loop will add 1
                except:
                    pass
        idx += 1
        
    # Deduplicate
    train_dict = {s[0]: s for s in train_samples}
    unique_train_samples = [train_dict[k] for k in sorted(train_dict.keys())]
    
    train_df = pd.DataFrame(unique_train_samples, columns=['id', 'x1', 'x2', 'x3', 'x4', 'd1', 'd2', 'd3'])
    train_df.to_csv('train.csv', index=False)
    
    print(f"Extracted {len(test_df)} test samples and {len(train_df)} train samples.")
