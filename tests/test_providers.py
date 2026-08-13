import unittest
from unittest.mock import MagicMock, patch

from supabase_easy_rag.providers.local_provider import CustomChatProvider, CustomEmbeddingProvider
from supabase_easy_rag.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider


class TestProviders(unittest.TestCase):
    def test_custom_embedding_provider(self):
        def dummy_embed(texts):
            return [[0.1, 0.2] for _ in texts]

        provider = CustomEmbeddingProvider(embed_fn=dummy_embed)
        self.assertEqual(provider.embed_query("test query"), [0.1, 0.2])
        self.assertEqual(provider.embed_texts(["t1", "t2"]), [[0.1, 0.2], [0.1, 0.2]])
        self.assertEqual(provider.embed_texts([]), [])

    def test_custom_chat_provider(self):
        def dummy_chat(prompt, system):
            return f"Echo: {prompt}"

        provider = CustomChatProvider(chat_fn=dummy_chat)
        res = provider.chat(prompt="hello")
        self.assertEqual(res, "Echo: hello")

    @patch("supabase_easy_rag.providers.openai.OpenAI")
    def test_openai_embedding_provider_mocked(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_embedding_item = MagicMock()
        mock_embedding_item.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_item]
        mock_instance.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider(api_key="sk-dummy", model="text-embedding-3-small")
        embeddings = provider.embed_texts(["hello world"])

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        mock_instance.embeddings.create.assert_called_once()

    @patch("supabase_easy_rag.providers.openai.OpenAI")
    def test_openai_chat_provider_mocked(self, mock_openai_cls):
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_choice = MagicMock()
        mock_choice.message.content = "Mocked LLM answer"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_response

        provider = OpenAIChatProvider(api_key="sk-dummy", model="gpt-4o-mini")
        answer = provider.chat(prompt="What is RAG?")

        self.assertEqual(answer, "Mocked LLM answer")
        mock_instance.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
