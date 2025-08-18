import logging

logger = logging.getLogger(__name__)

def build_smart_context_prompt(target_text: str, language: str, chunk_before: str = None, chunk_after: str = None) -> str:
    """Build prompt for ASR correction with duplicate removal"""
    has_before = chunk_before and chunk_before.strip()
    has_after = chunk_after and chunk_after.strip()

    if has_before and has_after:
        logger.info("Building MIDDLE chunk prompt (full context available)")
        context_info = f"Previous: {chunk_before.strip()}\nNext: {chunk_after.strip()}"
    elif has_before:
        logger.info("Building LAST chunk prompt (previous context only)")
        context_info = f"Previous: {chunk_before.strip()}"
    elif has_after:
        logger.info("Building FIRST chunk prompt (next context only)")
        context_info = f"Next: {chunk_after.strip()}"
    else:
        logger.info("Building SINGLE chunk prompt (no context available)")
        context_info = "No context available"

    # Build a much clearer and more explicit prompt
    prompt = f"""You are an ASR text correction expert. Your task is to fix ONLY the TARGET text by removing duplicate words that appear in PREVIOUS or NEXT chunks.

CRITICAL RULES:
1. ONLY output the corrected TARGET text - nothing else
2. NEVER add words from PREVIOUS or NEXT chunks
3. NEVER combine chunks together
4. REMOVE ONLY exact word duplicates that appear in PREVIOUS or NEXT chunks
5. Fix spelling mistakes in the TARGET text
6. Keep the original word order of the TARGET text
7. NEVER remove words unless they are EXACT duplicates

WHAT TO DO:
- REMOVE ONLY exact word duplicates that appear in PREVIOUS or NEXT chunks
- Fix spelling mistakes in the TARGET text
- Output ONLY the corrected TARGET text
- Be precise: only remove words that are truly duplicates

WHAT NOT TO DO:
- Do NOT add words from PREVIOUS or NEXT chunks
- Do NOT combine multiple chunks
- Do NOT add explanations or extra text
- Do NOT change the meaning of the TARGET text
- Do NOT remove words that are NOT exact duplicates
- Do NOT remove words just because they appear elsewhere

DUPLICATE VALIDATION RULES:
- A word is ONLY a duplicate if it appears EXACTLY the same in PREVIOUS or NEXT
- Different forms of the same word (singular/plural, different cases) are NOT duplicates
- Words that are part of different phrases are NOT duplicates
- Only remove words that are clearly redundant duplicates

EXAMPLES OF CORRECT DUPLICATE REMOVAL:
PREVIOUS: Hello my name is John
TARGET: my name is Sarah
NEXT: and I am a researcher
CORRECTED: Sarah
(Removed "my name is" because it appears exactly in PREVIOUS)

PREVIOUS: كيفية استخدام التكنولوجيا لحل
TARGET: لحل المشكلات اليومية
NEXT: وهذا ما دفعني إلى دراسة
CORRECTED: المشكلات اليومية
(Removed "لحل" because it appears exactly in PREVIOUS)

MULTIPLE DUPLICATES EXAMPLE:
PREVIOUS: كيفية استخدام التكنولوجيا لحل
TARGET: كيفية استخدام التكنولوجيا لحل
NEXT: لحل المشكلات اليومية
CORRECTED: استخدام التكنولوجيا
(Removed "كيفية" from PREVIOUS AND "لحل" from NEXT)

EXAMPLES OF WHAT NOT TO REMOVE:
PREVIOUS: اشتركوا في القناة
TARGET: على الكلام والترجمة الآلية
NEXT: None
CORRECTED: على الكلام والترجمة الآلية
(Keep "على" - it's NOT a duplicate with anything in PREVIOUS)

PREVIOUS: مرحبا اسمي عبدالعلي
TARGET: اسمي عبدالعزيز وانا باحث في مجال
NEXT: الزكاء الاصطناعي
CORRECTED: عبدالعزيز وانا باحث في مجال
(Remove "اسمي" - it appears exactly in PREVIOUS)

EXAMPLES OF SPELLING CORRECTION:
TARGET: منت سنوات كنت مهتما بكيفية
CORRECTED: منذ سنوات كنت مهتما بكيفية
(Fixed "منت" → "منذ" - common Arabic spelling mistake)

TARGET: الزكاء الاصطناعي
CORRECTED: الذكاء الاصطناعي
(Fixed "الزكاء" → "الذكاء" - common Arabic spelling mistake)

IMPORTANT: Be PRECISE about duplicates. Only remove words that are EXACTLY the same and clearly redundant. If you're unsure whether a word is a duplicate, KEEP it. It's better to keep a potential duplicate than to remove a word that isn't actually a duplicate.

PREVIOUS: {chunk_before.strip() if chunk_before else "None"}
TARGET: {target_text.strip()}
NEXT: {chunk_after.strip() if chunk_after else "None"}

Corrected TARGET:"""

    
    return prompt
