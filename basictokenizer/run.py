from basictokenizer import BasicTokenizer
import os

def main():
    # 1. Initialize the tokenizer
    tokenizer = BasicTokenizer()

    # 2. Load the Taylor Swift text file
    filename = "taylorswift.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, filename)
    print(f"Loading '{filename}'...")
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. Train the tokenizer
    # Let's do 14 merges to create a vocabulary of 270 tokens
    vocab_size = 270  
    print(f"\nTraining on {len(text)} characters of text...")
    tokenizer.train(text, vocab_size)

    # 4. Test the encode and decode round-trip!
    test_sentence = "taylor swift is writing a new song!"
    print(f"\nOriginal text: '{test_sentence}'")
    
    encoded = tokenizer.encode(test_sentence)
    print(f"Encoded tokens: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text:   '{decoded}'")
    
    # 5. Verify correctness
    if test_sentence == decoded:
        print("\n✅ Success! The round-trip decode perfectly matches the original text!")
    else:
        print("\n❌ Round-trip failed. Decoded text does not match.")

if __name__ == "__main__":
    main()
