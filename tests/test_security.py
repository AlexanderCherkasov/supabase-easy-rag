import unittest

from supabase_easy_rag.security.tokens import generate_secure_token, hash_token


class TestSecurity(unittest.TestCase):
    def test_token_generation_and_hashing(self):
        token = generate_secure_token(prefix="test_")
        self.assertTrue(token.startswith("test_"))
        self.assertGreater(len(token), 20)

        token_hash = hash_token(token)
        self.assertEqual(len(token_hash), 64)
        self.assertEqual(token_hash, hash_token(token))


if __name__ == "__main__":
    unittest.main()
