#!/usr/local/google/home/risapark/.venv/bin/python
import unittest
import sys
import os

# Ensure we can import from the current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tokenizer import BasicTokenizer

class TestBasicTokenizerEncode(unittest.TestCase):
    def test_encode_hello(self):
        """Test BPE encoding with predefined mock merge rules."""
        tokenizer = BasicTokenizer()
        
        # 1. Manually set up merge rules
        tokenizer.merges = {
            (104, 101): 256,  # 'h', 'e' -> 256
            (108, 108): 257,  # 'l', 'l' -> 257
            (257, 111): 258,  # 257, 'o' -> 258
        }
        
        # 2. Manually set up vocab dictionary
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        tokenizer.vocab[256] = b"he"
        tokenizer.vocab[257] = b"ll"
        tokenizer.vocab[258] = b"llo"

        # 3. Run encode
        encoded = tokenizer.encode("hello")
        
        # 4. Assert the output matches the expected list [256, 258]
        self.assertEqual(encoded, [256, 258])

if __name__ == "__main__":
    unittest.main()
