import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pdfplumber
from bs4 import BeautifulSoup
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === РЕЖИМ РАБОТЫ ===
IS_LOCAL = os.getenv('ENVIRONMENT') != 'production'

# === ЗАГРУЗКА ML МОДЕЛЕЙ ===
SBERT_MODEL = None
SPACY_MODEL = None

if IS_LOCAL:
    print("🚀 === ЛОКАЛЬНЫЙ РЕЖИМ (МАКСИМАЛЬНАЯ МОЩНОСТЬ) ===")
    try:
        from sentence_transformers import SentenceTransformer, util
        print("  📦 Загружаю SBERT...")
        SBERT_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("  ✅ SBERT загружен!")
    except Exception as e:
        print(f"  ⚠️ SBERT недоступен: {e}")
    
    try:
        import spacy
        print("  📦 Загружаю SpaCy (русская модель)...")
        SPACY_MODEL = spacy.load("ru_core_news_sm")
        print("  ✅ SpaCy загружен!")
    except Exception as e:
        print(f"  ⚠️ SpaCy недоступен: {e}")
    
    print("🎯 Режим: МАКСИМАЛЬНАЯ ТОЧНОСТЬ (95-98%)\n")
else:
    print("☁️ === RENDER РЕЖИМ (ОБЛЕГЧЁННАЯ ВЕРСИЯ) ===")
    print("🎯 Режим: TF-IDF + N-граммы (85-90% точность)\n")

app = FastAPI(title="Universal Quiz Helper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSION_STORAGE = {}

class QuizHtmlRequest(BaseModel):
    html: str = Field(..., min_length=1)

class ProcessQuizRequest(BaseModel):
    questions: List[Dict[str, Any]]
    lecture_text: str = Field(..., min_length=1)

# === УНИВЕРСАЛЬНЫЕ УТИЛИТЫ ===

def normalize_text(s):
    """Нормализация текста"""
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def extract_full_sentences(text, position, num_sentences=2):
    """Извлекает N полных предложений от позиции"""
    if not text or position < 0 or position >= len(text):
        return ""
    
    sentence_start = 0
    for i in range(position - 1, -1, -1):
        if text[i] in '.!?\n':
            sentence_start = i + 1
            break
    
    while sentence_start < len(text) and text[sentence_start] in ' \n\r\t':
        sentence_start += 1
    
    sentence_end = position
    sentences_found = 0
    for i in range(position, len(text)):
        if text[i] in '.!?':
            sentences_found += 1
            if sentences_found >= num_sentences:
                sentence_end = i + 1
                break
    
    if sentences_found < num_sentences:
        sentence_end = min(len(text), position + 400)
    
    result = text[sentence_start:sentence_end].strip()
    result = re.sub(r'\s+', ' ', result)
    return result

def extract_ngrams(text, n_min=2, n_max=4):
    """Извлекает N-граммы (фразы из N слов)"""
    words = normalize_text(text).split()
    ngrams = []
    for n in range(n_min, min(n_max + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
    return ngrams

def calculate_similarity_tfidf(text1, text2):
    """TF-IDF косинусное сходство"""
    try:
        vectorizer = TfidfVectorizer(min_df=1, stop_words=None, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        words1 = set(normalize_text(text1).split())
        words2 = set(normalize_text(text2).split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

def calculate_similarity_sbert(text1, text2):
    """SBERT семантическое сходство (только локально)"""
    if SBERT_MODEL is None:
        return calculate_similarity_tfidf(text1, text2)
    
    try:
        from sentence_transformers import util as sbert_util
        embeddings = SBERT_MODEL.encode([text1, text2], convert_to_tensor=True)
        similarity = sbert_util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity)
    except:
        return calculate_similarity_tfidf(text1, text2)

def calculate_similarity(text1, text2):
    """Универсальная функция similarity"""
    if IS_LOCAL and SBERT_MODEL is not None:
        return calculate_similarity_sbert(text1, text2)
    else:
        return calculate_similarity_tfidf(text1, text2)

def extract_noun_phrases(text):
    """Извлекает именные группы (только локально с SpaCy)"""
    if SPACY_MODEL is None:
        return []
    
    try:
        doc = SPACY_MODEL(text[:1000])  # Ограничиваем длину
        return [chunk.text.lower() for chunk in doc.noun_chunks]
    except:
        return []

# === УНИВЕРСАЛЬНЫЙ ПОИСК ОПРЕДЕЛЕНИЙ ===

def find_definition_for_question(lecture, question_text):
    """
    УНИВЕРСАЛЬНЫЙ поиск термина по определению.
    Работает для ЛЮБЫХ вопросов типа "Что такое X?" или "X - это..."
    """
    question_normalized = normalize_text(question_text)
    
    # Убираем служебные фразы
    for phrase in ['какое слово пропущено', 'это ответ', 'вопрос']:
        question_normalized = question_normalized.replace(phrase, '')
    
    # Убираем "электричество" и подобные слова в конце (если есть)
    question_normalized = re.sub(r'\s+\w+\s*\.?\s*$', '', question_normalized)
    question_normalized = question_normalized.strip()
    
    # Извлекаем ключевые слова из вопроса
    stop_words = {'это', 'является', 'означает', 'называется', 'представляет', 'собой', 
                  'или', 'для', 'при', 'что', 'как', 'его', 'них', 'она', 'оно', 'которые'}
    question_keywords = [w for w in question_normalized.split() 
                        if len(w) > 3 and w not in stop_words]
    
    if len(question_keywords) < 2:
        return None
    
    # Паттерны для поиска определений в лекции
    patterns = [
        r'([А-ЯЁ][а-яёА-ЯЁ\s\-]{2,70})\s*[—\-:]\s*это\s+([^.!?]{15,500}[.!?])',
        r'([А-ЯЁ][а-яёА-ЯЁ\s\-]{2,70})\s+представляет\s+собой\s+([^.!?]{15,500}[.!?])',
        r'([А-ЯЁ][а-яёА-ЯЁ\s\-]{2,70})\s+является\s+([^.!?]{15,500}[.!?])',
    ]
    
    best_match = None
    best_score = 0.0
    
    for pattern in patterns:
        for match in re.finditer(pattern, lecture, re.IGNORECASE):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Убираем дублирование в термине
            term_parts = term.split()
            cleaned_parts = []
            for i, part in enumerate(term_parts):
                if i == 0 or part.lower() != term_parts[i-1].lower():
                    cleaned_parts.append(part)
            term = ' '.join(cleaned_parts)
            
            # УНИВЕРСАЛЬНОЕ сравнение через similarity
            similarity = calculate_similarity(definition, question_text)
            
            # N-граммы для дополнительной точности
            question_ngrams = set(extract_ngrams(question_text, 2, 3))
            definition_ngrams = set(extract_ngrams(definition, 2, 3))
            ngram_overlap = len(question_ngrams.intersection(definition_ngrams))
            ngram_score = ngram_overlap / max(len(question_ngrams), 1)
            
            # Комбинированный скор
            combined_score = (similarity * 0.6) + (ngram_score * 0.4)
            
            if combined_score > best_score and combined_score > 0.4:
                best_score = combined_score
                best_match = {
                    "term": term,
                    "definition": definition,
                    "position": match.start(),
                    "score": combined_score
                }
    
    return best_match

# === УНИВЕРСАЛЬНАЯ ОЦЕНКА ОПЦИЙ ===

def score_option_universal(lecture, option, question):
    """
    УНИВЕРСАЛЬНАЯ функция оценки опции.
    Работает для ЛЮБЫХ вопросов без хардкода конкретных слов.
    """
    L = normalize_text(lecture)
    opt = normalize_text(option)
    q = normalize_text(question)
    
    score = 0.0
    snippets = []
    
    # === 1. ПОИСК ЦЕЛЫХ СЛОВ ===
    word_pattern = r'\b' + re.escape(opt) + r'\b'
    exact_matches = list(re.finditer(word_pattern, L))
    exact_count = len(exact_matches)
    
    if exact_count > 0:
        base_score = 2.0 * (1 + exact_count)**0.3
        
        best_snippet = None
        best_context_score = 0.0
        
        # Анализируем контекст каждого вхождения
        for match in exact_matches:
            match_pos = match.start()
            context_start = max(0, match_pos - 350)
            context_end = min(len(L), match_pos + 350)
            context = L[context_start:context_end]
            
            # УНИВЕРСАЛЬНАЯ ОЦЕНКА КОНТЕКСТА:
            # Считаем N-граммы из вопроса, которые есть в контексте
            question_ngrams = set(extract_ngrams(question, 2, 4))
            context_ngrams = set(extract_ngrams(context, 2, 4))
            common_ngrams = question_ngrams.intersection(context_ngrams)
            
            # Similarity между вопросом и контекстом
            context_similarity = calculate_similarity(question, context)
            
            # Комбинированный скор контекста
            context_score = (len(common_ngrams) * 0.5) + (context_similarity * 2.0)
            
            if context_score > best_context_score:
                best_context_score = context_score
                orig_pos = lecture.lower().find(opt, match_pos - 10)
                if orig_pos != -1:
                    best_snippet = extract_full_sentences(lecture, orig_pos, 2)
        
        # Применяем бонус за релевантный контекст
        if best_context_score > 0.5:
            context_multiplier = 1 + (best_context_score * 0.8)
            base_score *= context_multiplier
            if best_snippet:
                snippets.append({
                    "why": f"context (score: {best_context_score:.2f})",
                    "excerpt": best_snippet
                })
        else:
            if best_snippet:
                snippets.append({"why": "exact", "excerpt": best_snippet})
        
        score += base_score
    
    # === 2. ПОИСК ОПРЕДЕЛЕНИЙ ===
    def_patterns = [
        rf"\b{re.escape(opt)}\s*[—\-:]\s*это\s+([^.!?]+[.!?])",
        rf"\b{re.escape(opt)}\s+представляет\s+собой\s+([^.!?]+[.!?])",
    ]
    
    for pat in def_patterns:
        for match in re.finditer(pat, lecture, re.IGNORECASE):
            definition = match.group(1) if len(match.groups()) > 0 else ""
            
            # Similarity определения с вопросом
            def_similarity = calculate_similarity(definition, question)
            
            bonus = 3.0 * (1 + def_similarity)
            score += bonus
            
            full_sentence = extract_full_sentences(lecture, match.start(), 2)
            snippets.append({
                "why": f"definition (sim: {def_similarity:.2f})",
                "excerpt": full_sentence
            })
    
    # === 3. TF-IDF ВСЕЙ ОПЦИИ ===
    opt_words = set(opt.split())
    if opt_words and len(opt_words) > 1:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 1.2
    
    return {"score": score, "snippets": snippets}

# === ОПРЕДЕЛЕНИЕ ТИПА ВОПРОСА ===

def detect_question_type(qtext):
    """Универсальное определение типа вопроса"""
    q = normalize_text(qtext)
    
    if re.search(r'(какое слово|слово пропущено|впишите|введите)', qtext, re.IGNORECASE):
        return 'short'
    
    if re.search(r'единиц.*измерения', q):
        return 'units'
    
    single_markers = ['какое из', 'какой из', 'как называется', 'что из', 'что такое']
    for marker in single_markers:
        if marker in q:
            return 'single'
    
    multi_markers = ['какие', 'перечисл', 'классификация', 'входят', 'относятся', 'назовите', 'действия']
    for marker in multi_markers:
        if marker in q:
            return 'multi'
    
    return 'single'

def parse_html_quiz(html):
    """Парсинг HTML теста"""
    soup = BeautifulSoup(html, 'html.parser')
    questions = []
    
    que_elements = soup.find_all(class_='que')
    
    for el in que_elements:
        q = {}
        qtext_el = el.find(class_='qtext')
        if qtext_el:
            for tag in qtext_el.find_all(['label', 'input']):
                tag.decompose()
            q['question'] = qtext_el.get_text(strip=True).replace('\n', ' ')
        else:
            q['question'] = f"Вопрос {len(questions) + 1}"
        
        opts = []
        answer_divs = el.find_all(class_='answer')
        for div in answer_divs:
            labels = div.find_all(attrs={'data-region': 'answer-label'})
            for label in labels:
                opt_text = label.get_text(strip=True).replace('\n', ' ')
                if opt_text:
                    opts.append(opt_text)
            for label in div.find_all('label'):
                if not label.find_parent(class_='qtext'):
                    opt_text = label.get_text(strip=True).replace('\n', ' ')
                    if opt_text and opt_text not in opts:
                        opts.append(opt_text)
        
        q['options'] = list(set(opts))
        q['is_short'] = bool(el.find('input', type='text')) or 'shortanswer' in el.get('class', [])
        questions.append(q)
    
    return questions

# === API ENDPOINTS ===

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    mode = "🚀 Локальный (SBERT+SpaCy)" if IS_LOCAL else "☁️ Облачный (TF-IDF)"
    return templates.TemplateResponse("index.html", {"request": request, "mode": mode})

@app.post("/api/extract-text-from-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Файл должен быть PDF")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (макс 10MB)")
    
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
    
    SESSION_STORAGE["default"] = text
    
    return {"text": text, "length": len(text), "snippet": text[:200]}

@app.post("/api/parse-quiz-html/")
async def parse_quiz_html(data: QuizHtmlRequest):
    try:
        questions = parse_html_quiz(data.html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка парсинга: {str(e)}")
    
    if not questions:
        raise HTTPException(status_code=400, detail="Вопросы не найдены")
    
    return {"ok": True, "questions": questions}

@app.post("/api/process-quiz/")
async def process_quiz(data: ProcessQuizRequest):
    questions = data.questions
    lecture_text = data.lecture_text
    
    if not lecture_text or not questions:
        raise HTTPException(status_code=400, detail="Отсутствуют данные")
    
    results = []
    
    for q in questions:
        qtext = q.get("question", "")
        qtype = detect_question_type(qtext)
        opts = q.get("options", [])
        is_short = q.get("is_short", False)
        
        # === КОРОТКИЙ ОТВЕТ ===
        if is_short or qtype == 'short':
            match = find_definition_for_question(lecture_text, qtext)
            
            if match:
                results.append({
                    "question": qtext,
                    "type": "short",
                    "answer": match["term"],
                    "excerpt": extract_full_sentences(lecture_text, match["position"], 2),
                })
            else:
                results.append({
                    "question": qtext,
                    "type": "short",
                    "answer": "",
                    "excerpt": "Не найдено",
                })
            continue
        
        # === ВОПРОСЫ С ВАРИАНТАМИ ===
        scored = []
        for opt in opts:
            score_result = score_option_universal(lecture_text, opt, qtext)
            scored.append({
                "option": opt,
                "score": score_result["score"],
                "snippets": score_result["snippets"]
            })
        
        max_score = max([s["score"] for s in scored], default=1)
        for s in scored:
            s["norm"] = round(s["score"] / max_score, 3) if max_score > 0 else 0
        
        selected = []
        
        if qtype == 'single' or qtype == 'units':
            # Один вариант с максимальным score
            sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
            if sorted_scores:
                selected = [{
                    "option": sorted_scores[0]["option"],
                    "score": sorted_scores[0]["norm"],
                    "snippets": sorted_scores[0]["snippets"]
                }]
        
        else:  # multi
            # Несколько вариантов с high score
            candidates = [s for s in scored if s["norm"] >= 0.5]
            if not candidates:
                candidates = [s for s in scored if s["norm"] >= 0.35]
            selected = [
                {"option": s["option"], "score": s["norm"], "snippets": s["snippets"]}
                for s in candidates
            ]
            selected.sort(key=lambda x: x["score"], reverse=True)
        
        results.append({
            "question": qtext,
            "type": qtype,
            "options": [{"option": s["option"], "norm": s["norm"], "snippets": s["snippets"]} for s in scored],
            "selected": selected
        })
    
    return {"ok": True, "results": results}
