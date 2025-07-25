from utils import generate_prompts
from sklearn.model_selection import train_test_split

DATA_DIR = "data"

for i in range(1, 5):
    for rs in range(5):
        prompts = generate_prompts(10000, i, rs=rs, shuffle=True)

        # Generate train-test split
        train_prompts, test_prompts = train_test_split(prompts, test_size=0.2, random_state=42)

        # Save the splits to files
        with open(f'{DATA_DIR}/train_prompts_{i}_words_correlated_shuffled_rs{rs}.txt', 'w') as f:
            f.write('\n'.join(train_prompts))

        with open(f'{DATA_DIR}/test_prompts_{i}_words_correlated_shuffled_rs{rs}.txt', 'w') as f:
            f.write('\n'.join(test_prompts))

        print(f"Generated and saved train-test split for {i} words correlated")
        print(f"Train set size: {len(train_prompts)}")
        print(f"Test set size: {len(test_prompts)}")