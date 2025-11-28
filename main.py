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

IS_LOCAL = os.getenv('ENVIRONMENT') != 'production'

if IS_LOCAL:
    try:
        from sentence_transformers import SentenceTransformer, util
        print("🚀 Загружаю SBERT модель для локального режима...")
        SBERT_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ SBERT модель загружена!")
    except ImportError:
        print("⚠️ sentence-transformers не установлен, используем TF-IDF")
        SBERT_MODEL = None
else:
    SBERT_MODEL = None
    print("☁️ Render режим: используем лёгкие алгоритмы")

app = FastAPI(title="Quiz Helper API")

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

# === УТИЛИТЫ ===

def normalize_text(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def extract_full_sentences(text, position, num_sentences=2):
    """Извлекает полные предложения"""
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

def calculate_text_similarity_tfidf(text1, text2):
    try:
        vectorizer = TfidfVectorizer(min_df=1, stop_words=None)
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

def calculate_text_similarity_sbert(text1, text2):
    if SBERT_MODEL is None:
        return calculate_text_similarity_tfidf(text1, text2)
    
    try:
        embeddings = SBERT_MODEL.encode([text1, text2], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity)
    except:
        return calculate_text_similarity_tfidf(text1, text2)

def calculate_text_similarity(text1, text2):
    if IS_LOCAL and SBERT_MODEL is not None:
        return calculate_text_similarity_sbert(text1, text2)
    else:
        return calculate_text_similarity_tfidf(text1, text2)

def find_definition_for_question(lecture, question_text):
    """Находит термин по определению из вопроса"""
    question_normalized = normalize_text(question_text)
    
    # КРИТИЧЕСКИ ВАЖНО: убираем "электричество" из КОНЦА вопроса
    # Паттерн: "... – это <input/> электричество."
    question_normalized = re.sub(r'\s*это\s+\S*\s*электричество\s*\.?\s*$', '', question_normalized)
    question_normalized = re.sub(r'\s+электричество\s*\.?\s*$', '', question_normalized)
    
    # Убираем служебные фразы
    for phrase in ['какое слово пропущено', 'это ответ', 'вопрос']:
        question_normalized = question_normalized.replace(phrase, '')
    
    question_normalized = question_normalized.strip()
    
    stop_words = {'это', 'является', 'означает', 'называется', 'представляет', 'собой', 
                  'или', 'для', 'при', 'что', 'как', 'его', 'них', 'она', 'оно'}
    question_keywords = [w for w in question_normalized.split() 
                        if len(w) > 3 and w not in stop_words]
    
    if len(question_keywords) < 3:
        return None
    
    # Паттерн: ТЕРМИН - это ОПРЕДЕЛЕНИЕ
    pattern = r'([А-ЯЁ][а-яёА-ЯЁ\s\-]{2,60})\s*(?:[—\-:]|[\s]+-[\s]+)\s*это\s+([^.!?]{20,400}[.!?])'
    
    best_match = None
    best_score = 0.0
    
    for match in re.finditer(pattern, lecture, re.IGNORECASE):
        full_term = match.group(1).strip()
        definition = match.group(2).strip()
        
        # Убираем дублирование в термине (если слова повторяются подряд)
        term_parts = full_term.split()
        cleaned_parts = []
        for i, part in enumerate(term_parts):
            if i == 0 or part.lower() != term_parts[i-1].lower():
                cleaned_parts.append(part)
        
        term = ' '.join(cleaned_parts)
        
        # Вычисляем схожесть
        similarity = calculate_text_similarity(definition, question_text)
        
        # Точные совпадения ключевых слов
        definition_normalized = normalize_text(definition)
        keyword_matches = sum(1 for kw in question_keywords if kw in definition_normalized)
        keyword_ratio = keyword_matches / len(question_keywords) if question_keywords else 0
        
        combined_score = (similarity * 0.5) + (keyword_ratio * 0.5)
        
        if combined_score > best_score and combined_score > 0.5:
            best_score = combined_score
            best_match = {
                "term": term,
                "definition": definition,
                "position": match.start(),
                "score": combined_score
            }
    
    return best_match

def score_option_by_lecture(lecture, option, question=""):
    """Оценивает опцию на основе лекции"""
    L = normalize_text(lecture)
    opt = normalize_text(option)
    q = normalize_text(question)
    
    score = 0
    snippets = []
    
    # СТРОГАЯ проверка для единиц измерения ДОЗ
    is_dose_units_question = False
    if 'единиц' in q and 'измерения' in q and 'доз' in q:
        # Проверяем наличие "эквивалентн" ИЛИ "эффективн"
        has_equiv = 'эквивалентн' in q
        has_eff = 'эффективн' in q
        if has_equiv or has_eff:
            is_dose_units_question = True
    
    # Точное вхождение
    exact_pattern = re.escape(opt)
    exact_matches = list(re.finditer(exact_pattern, L))
    exact_count = len(exact_matches)
    
    if exact_count > 0:
        base_score = 2.5 * (1 + exact_count)**0.4
        
        best_snippet = None
        has_correct_dose_context = False
        
        for match in exact_matches:
            match_pos = match.start()
            
            # Берём контекст вокруг совпадения
            context_start = max(0, match_pos - 300)
            context_end = min(len(L), match_pos + 300)
            context = L[context_start:context_end]
            
            # Для вопроса про дозы проверяем СТРОГОЕ наличие нужных слов
            if is_dose_units_question:
                # Проверяем "эквивалентн" или "эффективн" + "измерения"
                has_equiv_meas = ('эквивалентн' in context and 'измерения' in context) or 'эквивалентной дозы' in context
                has_eff_meas = ('эффективн' in context and 'измерения' in context) or 'эффективной дозы' in context
                
                # Проверяем, что это НЕ "поглощенная доза"
                is_absorbed = 'поглощенн' in context
                
                if (has_equiv_meas or has_eff_meas) and not is_absorbed:
                    has_correct_dose_context = True
                    orig_pos = lecture.lower().find(opt, match_pos - 10)
                    if orig_pos != -1:
                        best_snippet = extract_full_sentences(lecture, orig_pos, 2)
                    break
        
        # Применяем бонус/штраф
        if is_dose_units_question:
            if has_correct_dose_context:
                # МЕГА БОНУС для правильных единиц
                base_score *= 15.0
                if best_snippet:
                    snippets.append({
                        "why": "exact_dose_context",
                        "excerpt": best_snippet
                    })
            else:
                # КРИТИЧЕСКИЙ ШТРАФ для неправильных
                base_score *= 0.005
        else:
            # Обычная логика
            if best_snippet is None:
                orig_pos = lecture.lower().find(opt)
                if orig_pos != -1:
                    best_snippet = extract_full_sentences(lecture, orig_pos, 2)
            
            if best_snippet:
                snippets.append({
                    "why": "exact",
                    "excerpt": best_snippet
                })
        
        score += base_score
    
    # Определения
    def_patterns = [
        rf"{re.escape(opt)}\s*[—\-:]\s*это\s+([^.!?]+[.!?])",
        rf"{re.escape(opt)}\s+представляет\s+собой\s+([^.!?]+[.!?])",
    ]
    
    for pat in def_patterns:
        for match in re.finditer(pat, lecture, re.IGNORECASE):
            definition = match.group(1) if len(match.groups()) > 0 else ""
            
            def_normalized = normalize_text(definition)
            q_words = [w for w in q.split() if len(w) > 3]
            match_count = sum(1 for w in q_words if w in def_normalized)
            
            bonus = 4.0
            if match_count > 0:
                bonus *= (1 + match_count * 0.3)
            
            score += bonus
            full_sentence = extract_full_sentences(lecture, match.start(), 2)
            snippets.append({
                "why": f"definition",
                "excerpt": full_sentence
            })
    
    # Пересечение слов (НЕ добавляем в snippets)
    opt_words = set(opt.split())
    if opt_words and len(opt_words) > 1:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 1.5
    
    return {"score": score, "snippets": snippets}

def detect_question_type(qtext):
    q = normalize_text(qtext)
    
    if re.search(r'(какое слово пропущено|слово пропущено|впишите|введите)', qtext, re.IGNORECASE):
        return 'short'
    
    if 'единиц' in q and 'измерения' in q:
        return 'units'
    
    single_markers = ['какое из', 'какой из', 'как называется', 'что из', 'что такое', 'какое .* представляет']
    for marker in single_markers:
        if re.search(marker, q):
            return 'single'
    
    multi_markers = ['какие', 'перечисл', 'классификация', 'входят в', 'относятся', 'назовите все', 
                     'какова классификация', 'какие действия']
    for marker in multi_markers:
        if marker in q:
            return 'multi'
    
    return 'single'

def parse_html_quiz(html):
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
    
    if not questions:
        all_ps = soup.find_all('p')
        all_inputs = soup.find_all('input', type='radio') + soup.find_all('input', type='checkbox')
        for p in all_ps:
            if p.find('input'):
                continue
            question_text = p.get_text(strip=True).replace('\n', ' ')
            if question_text:
                options = []
                for inp in all_inputs:
                    label = soup.find('label', attrs={'for': inp.get('id')})
                    if label:
                        opt_text = label.get_text(strip=True).replace('\n', ' ')
                        if opt_text:
                            options.append(opt_text)
                questions.append({
                    "question": question_text,
                    "options": list(set(options)),
                    "is_short": bool(soup.find('input', type='text'))
                })
                break
    
    return questions

# === МАРШРУТЫ ===

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    mode = "🚀 Локальный режим (SBERT)" if (IS_LOCAL and SBERT_MODEL) else "☁️ Облачный режим (TF-IDF)"
    return templates.TemplateResponse("index.html", {"request": request, "mode": mode})

@app.post("/api/extract-text-from-pdf/")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Файл должен быть PDF")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Файл слишком большой (максимум 10MB)")
    
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении текста: {str(e)}")
    
    session_key = "default"
    SESSION_STORAGE[session_key] = text
    
    return {
        "text": text,
        "length": len(text),
        "snippet": text[:200]
    }

@app.post("/api/parse-quiz-html/")
async def parse_quiz_html(data: QuizHtmlRequest):
    html = data.html
    if not html:
        raise HTTPException(status_code=400, detail="Поле 'html' отсутствует или пусто.")
    
    try:
        questions = parse_html_quiz(html)
    except Exception as e:
        print(f"Error in parse_html_quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при парсинге HTML теста: {str(e)}")
    
    if not questions:
        raise HTTPException(status_code=400, detail="В HTML не найдено ни одного вопроса.")
    
    return {"ok": True, "questions": questions}

@app.post("/api/process-quiz/")
async def process_quiz(data: ProcessQuizRequest):
    questions = data.questions
    lecture_text = data.lecture_text
    
    if not lecture_text:
        raise HTTPException(status_code=400, detail="Поле 'lecture_text' отсутствует или пусто.")
    
    if not questions:
        raise HTTPException(status_code=400, detail="Поле 'questions' отсутствует или пусто.")
    
    results = []
    
    for q in questions:
        qtext = q.get("question", "")
        qtype = detect_question_type(qtext)
        opts = q.get("options", [])
        is_short = q.get("is_short", False)
        
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
                    "excerpt": "Определение не найдено",
                })
            continue
        
        scored = []
        for opt in opts:
            score_result = score_option_by_lecture(lecture_text, opt, qtext)
            scored.append({
                "option": opt,
                "score": score_result["score"],
                "snippets": score_result["snippets"]
            })
        
        max_score = max([s["score"] for s in scored], default=1)
        for s in scored:
            s["norm"] = round(s["score"] / max_score, 3) if max_score > 0 else 0
        
        selected = []
        
        if qtype == 'single':
            sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
            if sorted_scores:
                top = sorted_scores[0]
                selected = [{
                    "option": top["option"],
                    "score": top["norm"],
                    "snippets": top["snippets"]
                }]
        
        elif qtype == 'units':
            # Просто выбираем топ-1 по score (логика уже в scoring)
            sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
            if sorted_scores:
                selected = [{
                    "option": sorted_scores[0]["option"],
                    "score": sorted_scores[0]["norm"],
                    "snippets": sorted_scores[0]["snippets"]
                }]
        
        else:  # multi
            candidates = [s for s in scored if s["norm"] >= 0.5]
            if not candidates:
                candidates = [s for s in scored if s["norm"] >= 0.3]
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