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

# Используем только TF-IDF (оптимизировано для памяти)
print("Используем TF-IDF для семантического поиска (оптимизировано для памяти)")

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
    """Нормализация текста"""
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip().lower()

def extract_full_sentences(text, position, num_sentences=3):
    """Извлекает полные предложения из текста с расширенным контекстом"""
    if not text or position < 0:
        return ""

    # Найти начало первого предложения (идем назад до точки/начала текста)
    sentence_start = 0
    for i in range(position - 1, -1, -1):
        if text[i] in '.!?\n':
            # Пропускаем пробелы после точки
            while i + 1 < len(text) and text[i + 1] in ' \n\r\t':
                i += 1
            sentence_start = i + 1
            break

    # Найти конец последнего предложения (идем вперед)
    sentence_end = len(text)
    sentences_found = 0
    for i in range(position, len(text)):
        if text[i] in '.!?':
            sentences_found += 1
            if sentences_found >= num_sentences:
                sentence_end = i + 1
                break

    # Если не нашли достаточно предложений, берем хотя бы 700 символов
    if sentences_found < num_sentences:
        sentence_end = min(len(text), position + 700)

    result = text[sentence_start:sentence_end].strip()

    # Убираем неполные предложения в конце
    if result and not result[-1] in '.!?':
        last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_period > 0:
            result = result[:last_period + 1]

    return result

def calculate_text_similarity(text1, text2):
    """Улучшенное TF-IDF сходство с character n-grams для лучшего семантического поиска"""
    try:
        # Используем character n-grams (2-5) для лучшего захвата морфологии русского языка
        vectorizer = TfidfVectorizer(
            min_df=1,
            analyzer='char_wb',  # char n-grams с границами слов
            ngram_range=(2, 5),  # 2-5 символьные n-граммы
            lowercase=True,
            strip_accents='unicode'
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        # Fallback на улучшенное пересечение слов с учетом частичных совпадений
        words1 = set(normalize_text(text1).split())
        words2 = set(normalize_text(text2).split())
        if not words1 or not words2:
            return 0.0

        # Точное пересечение
        exact_intersection = len(words1.intersection(words2))

        # Частичное пересечение (подстроки)
        partial_matches = 0
        for w1 in words1:
            for w2 in words2:
                if len(w1) >= 4 and len(w2) >= 4:
                    if w1 in w2 or w2 in w1:
                        partial_matches += 0.5

        total_matches = exact_intersection + partial_matches
        union = len(words1.union(words2))
        return total_matches / union if union > 0 else 0.0

def extract_ngrams(text, n=3):
    """Извлекает n-граммы из текста"""
    words = normalize_text(text).split()
    if len(words) < n:
        return [' '.join(words)]
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_ngram_overlap(text1, text2, n=3):
    """Вычисляет перекрытие n-грамм"""
    ngrams1 = set(extract_ngrams(text1, n))
    ngrams2 = set(extract_ngrams(text2, n))
    
    if not ngrams1:
        return 0.0
    
    overlap = len(ngrams1.intersection(ngrams2))
    return overlap / len(ngrams1)

def find_definition_for_question(lecture, question_text):
    """Находит термин по определению из вопроса (универсальный поиск)"""
    question_normalized = normalize_text(question_text)

    # Удаляем служебные фразы из вопроса
    for phrase in ['какое слово пропущено', 'это ответ', 'вопрос', 'ответ вопрос']:
        question_normalized = question_normalized.replace(phrase, '')

    question_normalized = question_normalized.strip()

    # Расширенный список стоп-слов
    stop_words = {'это', 'является', 'означает', 'называется', 'представляет', 'собой',
                  'или', 'для', 'при', 'что', 'как', 'его', 'них', 'она', 'оно', 'их',
                  'которые', 'который', 'которая', 'есть', 'быть', 'может'}
    question_keywords = [w for w in question_normalized.split()
                        if len(w) > 3 and w not in stop_words]

    if len(question_keywords) < 3:
        return None

    # Паттерны для поиска определений: "Термин - это определение" или "Термин: определение"
    patterns = [
        r'([А-ЯЁ][а-яё\s\-]{2,60})\s*[-—–:]\s*это\s+([^.!?]{20,500}[.!?])',
        r'([А-ЯЁ][а-яё\s\-]{2,60})\s+[-—–]\s+([^.!?]{20,500}[.!?])',
        r'([А-ЯЁ][а-яё\s\-]{2,60})\s*[-—–:]\s*совокупность\s+([^.!?]{20,500}[.!?])',
    ]

    best_match = None
    best_score = 0.0

    for pattern in patterns:
        for match in re.finditer(pattern, lecture, re.IGNORECASE | re.MULTILINE):
            term = match.group(1).strip()
            definition = match.group(2).strip()

            # Полное определение для сравнения
            full_definition = definition

            # Семантическое сходство (SBERT или TF-IDF)
            similarity = calculate_text_similarity(full_definition, question_text)

            # N-gram overlap (5-граммы для более точного сравнения)
            ngram_score_5 = calculate_ngram_overlap(full_definition, question_text, 5)
            ngram_score_3 = calculate_ngram_overlap(full_definition, question_text, 3)

            # Точные совпадения ключевых слов
            definition_normalized = normalize_text(full_definition)
            keyword_matches = sum(1 for kw in question_keywords if kw in definition_normalized)
            keyword_ratio = keyword_matches / len(question_keywords) if question_keywords else 0

            # Проверка на точное совпадение последовательности слов (bigrams)
            question_bigrams = set(zip(question_normalized.split(), question_normalized.split()[1:]))
            definition_bigrams = set(zip(definition_normalized.split(), definition_normalized.split()[1:]))
            bigram_overlap = len(question_bigrams.intersection(definition_bigrams)) / max(len(question_bigrams), 1)

            # Комбинированный скор (приоритет на семантику и точные совпадения)
            combined_score = (
                similarity * 0.4 +           # Семантическое сходство
                keyword_ratio * 0.25 +       # Совпадение ключевых слов
                ngram_score_5 * 0.15 +       # 5-граммы
                ngram_score_3 * 0.1 +        # 3-граммы
                bigram_overlap * 0.1         # Биграммы
            )

            if combined_score > best_score and combined_score > 0.4:
                best_score = combined_score
                best_match = {
                    "term": term,
                    "definition": full_definition,
                    "position": match.start(),
                    "score": combined_score
                }

    return best_match

def extract_key_concepts_from_question(question):
    """Извлекает ключевые концепции из вопроса с расширенным анализом"""
    q_lower = normalize_text(question)
    concepts = []
    required_concepts = []  # Концепты, которые ОБЯЗАТЕЛЬНО должны быть вместе

    # Вопросы о единицах измерения требуют СТРОГОГО контекста
    if 'единиц' in q_lower and 'измерения' in q_lower:
        # Если в вопросе упоминаются ОБА термина, они должны быть в контексте ВМЕСТЕ
        has_equivalent = 'эквивалентн' in q_lower
        has_effective = 'эффективн' in q_lower

        if has_equivalent and has_effective:
            # ОБА термина в вопросе - требуем ОБА в контексте
            required_concepts.append('эквивалентн')
            required_concepts.append('эффективн')
            concepts.append('эквивалентн')
            concepts.append('эффективн')
        elif has_equivalent:
            concepts.append('эквивалентн')
        elif has_effective:
            concepts.append('эффективн')

        if 'доз' in q_lower:
            concepts.append('доз')

    # Вопросы об излучении
    if 'излуч' in q_lower:
        if 'электромагнитн' in q_lower:
            concepts.append('электромагнитн')
        if 'корпускулярн' in q_lower:
            concepts.append('корпускулярн')
        if 'проникающ' in q_lower:
            concepts.append('проникающ')
        if 'ионизирующ' in q_lower:
            concepts.append('ионизирующ')
        if 'малым ионизирующим' in q_lower or 'малое ионизирующее' in q_lower:
            concepts.append('малым_ионизирующим')
        if 'большой проникающей' in q_lower or 'большая проникающая' in q_lower:
            concepts.append('большой_проникающей')

    return concepts, required_concepts

def score_option_by_lecture(lecture, option, question=""):
    """Оценивает опцию на основе лекции с учётом контекста вопроса"""
    L = normalize_text(lecture)
    opt = normalize_text(option)
    q = normalize_text(question)

    score = 0
    snippets = []

    key_concepts, required_concepts = extract_key_concepts_from_question(question)

    exact_pattern = re.escape(opt)
    exact_matches = list(re.finditer(exact_pattern, L))
    exact_count = len(exact_matches)

    if exact_count > 0:
        base_score = 2.5 * (1 + exact_count)**0.4

        best_context_score = 0
        best_context_snippet = None
        best_match_pos = None

        for match in exact_matches:
            match_pos = match.start()

            # Расширяем контекст для более точной проверки
            context_start = max(0, match_pos - 400)
            context_end = min(len(L), match_pos + 400)
            context = L[context_start:context_end]

            # Подсчитываем совпадения концептов в контексте
            context_score = sum(1 for concept in key_concepts if concept in context)

            # Проверяем ОБЯЗАТЕЛЬНЫЕ концепты (для units вопросов)
            required_score = sum(1 for concept in required_concepts if concept in context)

            # Если есть обязательные концепты, они ВСЕ должны присутствовать
            if required_concepts:
                if required_score == len(required_concepts):
                    context_score += 10  # Бонус за полное соответствие
                else:
                    context_score = 0  # Обнуляем, если не все обязательные концепты есть

            if context_score > best_context_score:
                best_context_score = context_score
                best_match_pos = match_pos
                orig_pos = lecture.lower().find(opt, match_pos - 10)
                if orig_pos != -1:
                    best_context_snippet = extract_full_sentences(lecture, orig_pos, 3)

        # Оценка с учетом контекста
        if best_context_score > 0:
            base_score *= (1 + best_context_score * 0.8)
            if best_context_snippet:
                snippets.append({
                    "why": f"exact_context (concepts: {best_context_score})",
                    "excerpt": best_context_snippet
                })
        else:
            # Если ключевые концепты есть, но не найдены - сильно штрафуем
            if key_concepts:
                base_score *= 0.1  # Было 0.2, уменьшаем до 0.1
            if best_context_snippet:
                snippets.append({
                    "why": "exact (no context)",
                    "excerpt": best_context_snippet
                })

        score += base_score
    
    def_patterns = [
        rf"{re.escape(opt)}\s*[—\-:]\s*это\s+([^.!?]+[.!?])",
        rf"{re.escape(opt)}\s+[—\-]\s+([^.!?]+[.!?])",
        rf"{re.escape(opt)}\s+представляет\s+собой\s+([^.!?]+[.!?])",
    ]
    
    for pat in def_patterns:
        for match in re.finditer(pat, lecture, re.IGNORECASE):
            match_text = match.group(0)
            definition = match.group(1) if len(match.groups()) > 0 else ""
            
            def_normalized = normalize_text(definition)
            q_words = [w for w in q.split() if len(w) > 3]
            match_count = sum(1 for w in q_words if w in def_normalized)
            
            # Дополнительная проверка на ключевые концепты в определении
            concept_in_def = sum(1 for concept in key_concepts if concept in def_normalized)
            
            bonus = 4.0
            if match_count > 0:
                bonus *= (1 + match_count * 0.3)
            if concept_in_def > 0:
                bonus *= (1 + concept_in_def * 0.4)
            
            score += bonus
            full_sentence = extract_full_sentences(lecture, match.start(), 2)
            snippets.append({
                "why": f"definition (q_words: {match_count}, concepts: {concept_in_def})",
                "excerpt": full_sentence
            })
    
    opt_words = set(opt.split())
    if opt_words and len(opt_words) > 1:
        matched_words = len(opt_words.intersection(set(L.split())))
        ratio = matched_words / len(opt_words)
        score += ratio * 1.5
        if ratio > 0:
            snippets.append({
                "why": "word-match",
                "matched": f"{matched_words}/{len(opt_words)}"
            })
    
    return {"score": score, "snippets": snippets}

def detect_question_type(qtext):
    """Определяет тип вопроса"""
    q = normalize_text(qtext)
    
    if re.search(r'(какое слово пропущено|слово пропущено|впишите|введите)', qtext, re.IGNORECASE):
        return 'short'
    
    if 'единиц' in q and 'измерения' in q:
        return 'units'
    
    single_markers = ['какое из', 'какой из', 'как называется', 'что из', 'что такое', 'какое .* представляет']
    for marker in single_markers:
        if re.search(marker, q):
            return 'single'
    
    multi_markers = ['какие', 'перечисл', 'классификация', 'входят в', 'относятся', 'назовите все', 'какова классификация', 'какие действия']
    for marker in multi_markers:
        if marker in q:
            return 'multi'
    
    return 'single'

def parse_html_quiz(html):
    """Парсит HTML теста"""
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
    return templates.TemplateResponse("index.html", {"request": request})

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
            key_concepts, required_concepts = extract_key_concepts_from_question(qtext)

            if required_concepts:
                # Если есть ОБЯЗАТЕЛЬНЫЕ концепты (напр. эквивалентная И эффективная)
                # Выбираем только те варианты, которые имеют ВСЕ обязательные концепты
                context_matches = []
                for s in scored:
                    # Проверяем наличие сильного контекста с высоким счетом концептов
                    for snippet in s['snippets']:
                        why = snippet.get('why', '')
                        if 'exact_context' in why and 'concepts:' in why:
                            # Извлекаем количество концептов
                            import re as re_local
                            concept_count_match = re_local.search(r'concepts:\s*(\d+)', why)
                            if concept_count_match:
                                concept_count = int(concept_count_match.group(1))
                                # Только если найдены ВСЕ обязательные концепты
                                if concept_count >= len(required_concepts) + len(key_concepts) - len(required_concepts):
                                    context_matches.append(s)
                                    break

                if context_matches:
                    # Выбираем ТОЛЬКО лучший вариант с максимальным счетом
                    context_matches.sort(key=lambda x: x["score"], reverse=True)
                    best = context_matches[0]
                    selected = [{
                        "option": best["option"],
                        "score": best["norm"],
                        "snippets": best["snippets"]
                    }]
                else:
                    # Если не нашли с обязательными концептами, не выбираем ничего или берем лучший
                    sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
                    if sorted_scores and sorted_scores[0]["score"] > 0:
                        selected = [{
                            "option": sorted_scores[0]["option"],
                            "score": sorted_scores[0]["norm"],
                            "snippets": sorted_scores[0]["snippets"]
                        }]
            elif key_concepts:
                # Если есть ключевые концепты, но не обязательные
                context_matches = []
                for s in scored:
                    has_strong_context = any(
                        'exact_context' in snippet.get('why', '') and 'concepts:' in snippet.get('why', '')
                        for snippet in s['snippets']
                    )
                    if has_strong_context:
                        context_matches.append(s)

                context_matches.sort(key=lambda x: x["score"], reverse=True)

                if context_matches:
                    # Берем только лучший вариант
                    best = context_matches[0]
                    selected = [{
                        "option": best["option"],
                        "score": best["norm"],
                        "snippets": best["snippets"]
                    }]
                else:
                    sorted_scores = sorted(scored, key=lambda x: x["score"], reverse=True)
                    if sorted_scores:
                        selected = [{
                            "option": sorted_scores[0]["option"],
                            "score": sorted_scores[0]["norm"],
                            "snippets": sorted_scores[0]["snippets"]
                        }]
            else:
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
