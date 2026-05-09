from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskKind = Literal["article", "story", "caption", "headline", "rewrite", "summary", "translation", "general"]


@dataclass(frozen=True)
class RequestAnalysis:
    task_kind: TaskKind
    needs_search: bool
    reason: str


FRESHNESS_KEYWORDS = [
    # English
    "latest", "today", "yesterday", "current", "currently", "new", "recent",
    "news", "breaking", "now", "this week", "this month", "this year",
    "price", "cost", "release date", "launch date", "box office", "grossed",
    "tracking", "opening weekend", "score", "schedule", "stock", "market",
    "2025", "2026", "2027",
    # Arabic
    "أحدث", "آخر", "اليوم", "أمس", "حاليًا", "حاليا", "الجديد", "خبر",
    "أخبار", "عاجل", "هذا الأسبوع", "هذا الشهر", "هذا العام", "سعر", "تكلفة",
    "موعد الإصدار", "تاريخ الإصدار", "إيرادات", "شباك التذاكر", "افتتاحية",
    "حقق", "تتبع", "الأسواق", "البورصة",
]

REAL_WORLD_ENTITIES = [
    "openai", "google", "microsoft", "apple", "meta", "tesla", "nvidia",
    "netflix", "disney", "warner", "amazon", "anthropic", "xai", "deepseek",
    "saudi", "uae", "egypt", "qatar", "china", "us", "uk",
    "أوبن إيه آي", "جوجل", "مايكروسوفت", "آبل", "ميتا", "تسلا", "إنفيديا",
    "نتفليكس", "ديزني", "وارنر", "أمازون", "أنثروبيك", "ديب سيك",
    "السعودية", "الإمارات", "مصر", "قطر", "الصين", "أميركا", "أمريكا",
]

NO_SEARCH_INTENT_KEYWORDS = [
    "rewrite", "paraphrase", "summarize", "summary", "translate", "proofread",
    "make it shorter", "caption from this", "صياغة", "أعد صياغة", "تلخيص", "لخص",
    "ترجمة", "ترجم", "تدقيق", "صحح", "اختصر", "بوست من النص", "من هذا النص",
]

STORY_KEYWORDS = ["story", "novel", "scene", "script", "حكاية", "قصة", "رواية", "سيناريو", "مشهد"]
ARTICLE_KEYWORDS = ["article", "report", "news", "blog", "seo", "مقال", "تقرير", "خبر", "صحفي", "سيو"]
CAPTION_KEYWORDS = ["caption", "tweet", "x post", "social", "منصة إكس", "تغريدة", "كابشن", "بوست"]
HEADLINE_KEYWORDS = ["headline", "title", "عنوان", "عناوين"]
SUMMARY_KEYWORDS = ["summarize", "summary", "تلخيص", "لخص"]
TRANSLATION_KEYWORDS = ["translate", "translation", "ترجمة", "ترجم"]
REWRITE_KEYWORDS = ["rewrite", "paraphrase", "proofread", "أعد صياغة", "صياغة", "تدقيق", "صحح"]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(k.lower() in text for k in keywords)


def detect_task_kind(user_text: str) -> TaskKind:
    text = user_text.lower()
    if _contains_any(text, TRANSLATION_KEYWORDS):
        return "translation"
    if _contains_any(text, SUMMARY_KEYWORDS):
        return "summary"
    if _contains_any(text, REWRITE_KEYWORDS):
        return "rewrite"
    if _contains_any(text, HEADLINE_KEYWORDS):
        return "headline"
    if _contains_any(text, CAPTION_KEYWORDS):
        return "caption"
    if _contains_any(text, STORY_KEYWORDS):
        return "story"
    if _contains_any(text, ARTICLE_KEYWORDS):
        return "article"
    return "general"


def analyze_request(user_text: str) -> RequestAnalysis:
    text = user_text.lower().strip()
    task_kind = detect_task_kind(text)

    has_fresh_keyword = _contains_any(text, FRESHNESS_KEYWORDS)
    has_real_entity = _contains_any(text, REAL_WORLD_ENTITIES)
    has_no_search_intent = _contains_any(text, NO_SEARCH_INTENT_KEYWORDS)

    # Creative tasks do not need search unless the user explicitly asks for real/current facts.
    if task_kind == "story" and not has_fresh_keyword:
        return RequestAnalysis(task_kind, False, "creative_story_no_fresh_data_needed")

    # Rewrite/summary/translation usually should use only provided text unless user asks to verify/update facts.
    if task_kind in {"rewrite", "summary", "translation"} and not has_fresh_keyword:
        return RequestAnalysis(task_kind, False, "transform_existing_text_only")

    if has_fresh_keyword:
        return RequestAnalysis(task_kind, True, "freshness_keyword_detected")

    # Real entities in short prompts usually need grounding because the user may expect factual accuracy.
    if has_real_entity and len(text) < 1800 and not has_no_search_intent:
        return RequestAnalysis(task_kind, True, "real_world_entity_detected")

    return RequestAnalysis(task_kind, False, "no_external_fresh_data_required")


def resolve_search_enabled(user_text: str, search_mode: str = "auto") -> tuple[bool, RequestAnalysis]:
    analysis = analyze_request(user_text)

    if search_mode == "on":
        return True, analysis
    if search_mode == "off":
        return False, analysis
    return analysis.needs_search, analysis


def temperature_for_task(task_kind: TaskKind, uses_search: bool) -> float:
    if uses_search:
        return 0.35

    return {
        "story": 0.85,
        "article": 0.55,
        "caption": 0.70,
        "headline": 0.75,
        "rewrite": 0.35,
        "summary": 0.25,
        "translation": 0.30,
        "general": 0.45,
    }.get(task_kind, 0.45)


def max_tokens_for_task(task_kind: TaskKind) -> int:
    return {
        "story": 4000,
        "article": 3500,
        "caption": 700,
        "headline": 700,
        "rewrite": 1800,
        "summary": 1200,
        "translation": 1600,
        "general": 2000,
    }.get(task_kind, 2000)
