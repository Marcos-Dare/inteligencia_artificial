import re

if __name__ == '__main__':
    with open('PMC1_content.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    test_samples = []
    train_samples = []
    
    # We know test samples are exactly 1 to 20 right after yrede (T5)
    # The first '1' is at line 63. Let's just find the first '1' that is followed by 4 floats.
    
    idx = 0
    # Find test data
    in_test = False
    for i, line in enumerate(lines):
        if line == 'yrede' and lines[i+1] == '(T5)':
            idx = i + 2
            break
            
    for i in range(1, 21):
        while idx < len(lines):
            if lines[idx] == str(i):
                try:
                    x1 = float(lines[idx+1])
                    x2 = float(lines[idx+2])
                    x3 = float(lines[idx+3])
                    d = float(lines[idx+4])
                    test_samples.append((x1, x2, x3, d))
                    idx += 5
                    break
                except:
                    idx += 1
            else:
                idx += 1
                
    with open('test.csv', 'w') as f:
        f.write("x1,x2,x3,d\n")
        for s in test_samples:
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]}\n")
            
    # Find train data
    # The train data follows 'ANEXO'.
    anexo_idx = 0
    for i, line in enumerate(lines):
        if line == 'ANEXO':
            anexo_idx = i
            break
            
    idx = anexo_idx
    while idx < len(lines):
        # try to match sample ID 1 to 200
        if re.match(r'^\d+$', lines[idx]):
            sample_id = int(lines[idx])
            if 1 <= sample_id <= 200:
                try:
                    x1 = float(lines[idx+1])
                    x2 = float(lines[idx+2])
                    x3 = float(lines[idx+3])
                    d = float(lines[idx+4])
                    train_samples.append((sample_id, x1, x2, x3, d))
                    idx += 4 # skip the 4 floats, the loop will +=1 to skip the current
                except:
                    pass
        idx += 1
        
    # We might have duplicates if test samples and train samples have same id?
    # No, train_samples only parsed after ANEXO.
    # Keep only unique IDs by using a dict
    train_dict = {s[0]: s for s in train_samples}
    unique_train_samples = [train_dict[k] for k in sorted(train_dict.keys())]
    
    with open('train.csv', 'w') as f:
        f.write("id,x1,x2,x3,d\n")
        for s in unique_train_samples:
            f.write(f"{s[0]},{s[1]},{s[2]},{s[3]},{s[4]}\n")
            
    print(f"Extracted {len(test_samples)} test samples and {len(unique_train_samples)} train samples.")
