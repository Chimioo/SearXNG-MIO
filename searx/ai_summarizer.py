# SPDX-License-Identifier: AGPL-3.0-or-later
"""AI summarizer module for SearXNG"""

import json
import os
import re
import typing as t
from typing import Generator, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from searx import logger, settings

logger = logger.getChild('ai_summarizer')


class AISummarizer:
    """AI summarizer using OpenAI compatible APIs"""

    def __init__(self):
        self.enabled = False
        self.client: Optional[OpenAI] = None
        self.model = "gpt-3.5-turbo"
        self.max_tokens = 1000
        self.temperature = 0.7
        self.api_base = "https://api.openai.com/v1"
        self.timeout = 30
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client from settings"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not installed. AI summarizer disabled.")
            return

        ai_settings = settings.get('ai', {})
        if not ai_settings.get('enabled', False):
            logger.info("AI summarizer is disabled in settings.")
            return

        api_key = ai_settings.get('api_key', '')
        if not api_key:
            logger.warning("No API key provided for AI summarizer.")
            return

        self.enabled = True
        self.model = ai_settings.get('model', self.model)
        self.max_tokens = ai_settings.get('max_tokens', self.max_tokens)
        self.temperature = ai_settings.get('temperature', self.temperature)
        self.api_base = ai_settings.get('api_base', self.api_base)
        self.timeout = ai_settings.get('timeout', self.timeout)
        self.max_results = ai_settings.get('max_results', 10)
        
        api_key = ai_settings.get('api_key', '')
        if api_key.startswith('{{env') and '}}' in api_key:
            env_var = re.search(r'{{env\s+(\w+)}}', api_key).group(1)
            api_key = os.environ.get(env_var, '')
        self.api_key = api_key

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            logger.info("AI summarizer initialized with model: %s", self.model)
        except Exception as e:
            logger.error("Failed to initialize OpenAI client: %s", str(e))
            self.enabled = False

    def _build_prompt(self, query: str, results: t.List[t.Dict], language: str = 'en') -> str:
        """Build prompt for AI summarization"""
        results_text = []
        for i, result in enumerate(results[:self.max_results], 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            img = result.get('img', '')
            img_info = f"\nImage URL: {img}" if img else ''
            results_text.append(f"<result id=\"{i}\">\nTitle: {title}\nContent: {content}{img_info}\n</result>\n")

        results_str = "\n".join(results_text)
        
        prompt = f"""You are a helpful search assistant for SearXNG. Summarize the following search results for the query: "{query}"

Search Results:
{results_str}

IMPORTANT: Respond in the following language: {language}
Use the <result id="X"> id attribute to reference sources in your <cite index="X"></cite> tags

Format your summary with clickable citations using this EXACT XML format:
- Use **bold** for key terms and important facts
- Use bullet points (-) for lists when appropriate
- Use `inline code` for technical terms or commands
- Cite sources using <cite index="1"></cite>, <cite index="2"></cite>, etc. RIGHT AFTER the information you're citing
- Place the <cite> tag immediately after the fact or claim it supports, before any punctuation or space
- Example: "Python was created in 1991<cite index="1"></cite> and is widely used today<cite index="2"></cite>."
- Each result has an "img" field with image URL, use it ONLY when visual adds significant value
- Embed images using <img> tag: <img src="full_image_url" /> - place on separate lines
- Use images sparingly - only when truly helpful for understanding (e.g., people, products, visual concepts)
- Keep paragraphs short and well-organized

Summary:"""

        return prompt

    def summarize_stream(self, query: str, results: t.List[t.Dict], language: str = 'en') -> Generator[str, None, None]:
        """Generate streaming summary"""
        if not self.enabled or not self.client:
            yield "AI summarizer is not available. Please check your settings."
            return

        if not results:
            yield "No search results to summarize."
            return

        try:
            prompt = self._build_prompt(query, results, language)
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a helpful search assistant for SearXNG that summarizes search results clearly and accurately, citing sources appropriately. Always respond in the language: {language}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("AI summarization error: %s", str(e))
            yield f"Error generating summary: {str(e)}"

    def summarize(self, query: str, results: t.List[t.Dict], language: str = 'en') -> str:
        """Generate non-streaming summary"""
        return ''.join(self.summarize_stream(query, results, language))


# Global instance
_summarizer: Optional[AISummarizer] = None


def get_summarizer() -> AISummarizer:
    """Get or create the AI summarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = AISummarizer()
    return _summarizer


def is_available() -> bool:
    """Check if AI summarizer is available"""
    return get_summarizer().enabled
