from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="예상 질문 생성기", description="PostgreSQL 테이블 스키마 정보를 기반으로 예상 질문 목록을 생성합니다.")

# 데이터베이스 연결 설정
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

# Ollama 서버 설정
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")

# nl.json 파일 경로 (환경 변수로 설정하거나 기본값 사용)
NL_JSON_PATH = os.getenv("NL_JSON_PATH", "nl.json")

# 모델 정의
class TableInfo(BaseModel):
    table_name: str
    limit_unique_values: int = Field(10, description="각 컬럼당 가져올 유니크 값의 최대 개수")
    
class QuestionGenRequest(BaseModel):
    table_name: str
    example_questions: List[str] = Field(..., description="사용자 질문 예시 목록")
    num_questions: int = Field(10, description="생성할 질문 개수")

class EnhancedQuestion(BaseModel):
    original_question: str
    final_question: str
    tag: str
    score: Optional[float] = None

class QuestionGenResponse(BaseModel):
    generated_questions: List[EnhancedQuestion]
    table_schema: Dict[str, Any]
    
# nl.json 파일 로드 함수
def load_nl_json() -> List[Dict[str, str]]:
    """nl.json 파일을 로드하여 질문 예시와 태그 정보를 반환"""
    try:
        with open(NL_JSON_PATH, 'r', encoding='utf-8') as f:
            nl_data = json.load(f)
        return nl_data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"nl.json 파일 로드 오류: {str(e)}")
        # 기본 예시 데이터 반환 (실제로는 nl.json 파일이 필요)
        return [
            {
                "original_question": "2021년w45 주요 지표 요약해줘",
                "final_question": "2021년45주차의 주요 지표 현황을 요약해줘.",
                "tag": "getMainIndexState"
            },
            {
                "original_question": "2021년 44주차 주요 지표를 요약해줘",
                "final_question": "2021년 44주차의 주요 지표 현황을 요약해줘.",
                "tag": "getMainIndexState"
            }
        ]

# 유효한 태그 목록 추출 함수
def get_valid_tags() -> List[str]:
    """nl.json에서 유효한 태그 목록을 추출"""
    nl_data = load_nl_json()
    valid_tags = set()
    
    for item in nl_data:
        if "tag" in item and item["tag"]:
            valid_tags.add(item["tag"])
    
    return list(valid_tags)
    
async def get_ollama_completion(prompt: str) -> str:
    """Ollama API를 통해 텍스트 생성"""
    url = f"{OLLAMA_URL}/api/generate"
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Ollama 서버 오류: {str(e)}")

def get_db_connection():
    """PostgreSQL 데이터베이스 연결"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"데이터베이스 연결 오류: {str(e)}")

def get_column_comments(table_name: str) -> Dict[str, str]:
    """테이블 컬럼의 코멘트 정보를 가져옴"""
    conn = get_db_connection()
    comments = {}
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT a.attname as column_name,
                       pg_catalog.col_description(a.attrelid, a.attnum) as column_comment
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = %s
                  AND a.attnum > 0
                  AND NOT a.attisdropped;
            """, (table_name,))
            
            for row in cursor.fetchall():
                if row['column_comment']:
                    comments[row['column_name']] = row['column_comment']
    
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"컬럼 코멘트 가져오기 오류: {str(e)}")
    finally:
        conn.close()
        
    return comments

def get_table_schema(table_name: str) -> Dict[str, Any]:
    """테이블의 스키마 정보를 가져옴"""
    conn = get_db_connection()
    result = {}
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # 테이블 존재 확인
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))
            
            if not cursor.fetchone()['exists']:
                raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'이 존재하지 않습니다.")
            
            # 컬럼 정보 가져오기
            cursor.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cursor.fetchall()
            result["columns"] = [dict(col) for col in columns]
            
            # 테이블 통계 정보
            cursor.execute(f"SELECT COUNT(*) as row_count FROM {table_name};")
            result["row_count"] = cursor.fetchone()['row_count']
            
            # 기본 키 정보
            cursor.execute("""
                SELECT c.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                  AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
            """, (table_name,))
            
            result["primary_keys"] = [row['column_name'] for row in cursor.fetchall()]
            
            # 컬럼 코멘트 정보 가져오기
            column_comments = get_column_comments(table_name)
            
            # 컬럼 정보에 코멘트 추가
            for column in result["columns"]:
                column_name = column["column_name"]
                if column_name in column_comments:
                    column["comment"] = column_comments[column_name]
                else:
                    column["comment"] = ""
            
        return result
    
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"스키마 정보 가져오기 오류: {str(e)}")
    finally:
        conn.close()

def get_unique_values(table_name: str, column_name: str, data_type: str, limit: int) -> List[str]:
    """숫자를 제외한 컬럼의 유니크 값을 가져옴"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 숫자 타입은 제외
            if data_type in ('integer', 'bigint', 'smallint', 'decimal', 'numeric', 'real', 'double precision'):
                return []
            
            query = f"""
                SELECT DISTINCT {column_name} 
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                ORDER BY {column_name}
                LIMIT %s;
            """
            
            cursor.execute(query, (limit,))
            values = [str(row[0]) for row in cursor.fetchall()]
            return values
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"유니크 값 가져오기 오류: {str(e)}")
    finally:
        conn.close()

async def evaluate_questions(questions: List[EnhancedQuestion], schema_info: Dict, valid_tags: List[str], nl_examples: List[Dict] = None) -> List[Dict]:
    """질문을 평가하고 점수와 개선 피드백을 제공 - nl.json 패턴을 고려한 개선 버전"""
    
    # nl.json 질문 패턴 추출 (제공된 경우)
    patterns = []
    if nl_examples:
        for example in nl_examples:
            if "original_question" in example and "final_question" in example:
                patterns.append({
                    "original": example["original_question"],
                    "final": example["final_question"]
                })
    
    evaluation_prompt = """
당신은 데이터 분석 질문 평가 전문가입니다. 다음 질문 목록을 평가하고 각 질문에 0.0에서 1.0 사이의 점수를 매겨주세요.

평가 기준:
1. 명확성: 질문이 명확하고 이해하기 쉬운가?
2. 관련성: 질문이 주어진 데이터 스키마와 관련이 있는가?
3. 구체성: 질문이 필요한 정보를 충분히 구체적으로 요청하는가?
4. 실행 가능성: 이 질문은 주어진 데이터로 답변 가능한가?
5. 자연스러움: 질문이 자연스러운 대화체로 표현되었는가?
6. 태그 적절성: 질문에 할당된 태그가 유효하고 적절한가? (유효한 태그: {})
7. 패턴 일치성: final_question이 nl.json의 질문 패턴을 따르고 있는가?

nl.json 질문 패턴 예시:
{}

테이블 스키마 정보:
{}

질문 목록:
{}

특히 아래 패턴을 확인하세요:
1. 주차 표기 방식: "2023년 52W" -> "2023년 52주차에서"
2. 국가명 표기 방식: "브라질" -> "Brazil 국가의"
3. 문장 종결 방식: "~은 얼마인가?" -> "~을 알려줘." 또는 "~을 요약해줘."

각 질문에 대해 JSON 배열 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요.
[
  {{
    "index": 0, 
    "score": 0.85, 
    "feedback": "피드백 내용", 
    "suggested_tag": "적절한 태그",
    "pattern_match": "패턴 일치 여부에 대한 피드백",
    "improved_final_question": "nl.json 패턴에 맞게 개선된 최종 질문"
  }},
  ...
]
""".format(
        json.dumps(valid_tags, ensure_ascii=False),
        json.dumps(patterns, ensure_ascii=False, indent=2),
        json.dumps(schema_info, ensure_ascii=False, indent=2),
        json.dumps([{"original_question": q.original_question, "final_question": q.final_question, "tag": q.tag} for q in questions], ensure_ascii=False, indent=2)
    )
    
    response = await get_ollama_completion(evaluation_prompt)
    
    try:
        # 텍스트에서 JSON 배열 추출하기
        import re
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                evaluation_results = json.loads(json_str)
                return evaluation_results
            except json.JSONDecodeError:
                pass
        
        # 정규식으로 찾지 못한 경우 다른 방법 시도
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            try:
                evaluation_results = json.loads(json_str)
                return evaluation_results
            except json.JSONDecodeError:
                pass
                
        # 위 방법들로 파싱에 실패한 경우 기본 점수 반환
        return [{"index": i, "score": 0.75, "feedback": "질문의 품질은 양호하지만 개선의 여지가 있습니다.", "suggested_tag": q.tag if q.tag in valid_tags else valid_tags[0]} for i, q in enumerate(questions)]
    except Exception as e:
        # 모든 예외 처리 (JSON 파싱 실패 등)
        print(f"평가 프로세스 중 오류 발생: {str(e)}")
        return [{"index": i, "score": 0.75, "feedback": "평가 프로세스 중 오류가 발생했습니다.", "suggested_tag": q.tag if q.tag in valid_tags else valid_tags[0]} for i, q in enumerate(questions)]


async def improve_question(question: EnhancedQuestion, feedback: str, schema_info: Dict, valid_tags: List[str], nl_examples: List[Dict] = None) -> EnhancedQuestion:
    """피드백을 기반으로 질문 개선 - nl.json 패턴을 고려한 개선 버전"""
    
    # nl.json 질문 패턴 추출 (제공된 경우)
    patterns = []
    if nl_examples:
        for example in nl_examples:
            if "original_question" in example and "final_question" in example:
                patterns.append({
                    "original": example["original_question"],
                    "final": example["final_question"]
                })
    
    improvement_prompt = """
당신은 데이터 분석 질문 개선 전문가입니다. 다음 질문을 주어진 피드백에 따라 개선하고, nl.json의 질문 패턴을 따르도록 만들어주세요.

테이블 스키마 정보:
{}

nl.json 질문 패턴 예시:
{}

원본 질문: {}
최종 질문: {}
현재 태그: {}

피드백: {}

유효한 태그 목록: {}

개선 지침:
1. original_question은 유지하고, final_question만 개선하세요.
2. 주차 표기 방식: "2023년 52W" -> "2023년 52주차에서"
3. 국가명 표기 방식: "브라질" -> "Brazil 국가의" 
4. 문장 종결 방식: "~은 얼마인가?" -> "~을 알려줘." 또는 "~을 요약해줘."
5. nl.json 패턴에 맞게 final_question을 재구성하세요.

예를 들어:
- original_question: "2023년 52주차 브라질 법인의 매출액은 얼마인가?"
- 잘못된 final_question: "2023년 52주차 브라질 법인의 매출액을 알려줘."
- 올바른 final_question: "2023년 52주차에서 Brazil 국가의 매출액 현황을 알려줘."

개선된 질문을 다음 형식의 JSON 객체로만 응답해주세요. 다른 텍스트는 포함하지 마세요.
{{
  "original_question": "원본 질문을 그대로 유지",
  "final_question": "개선된 최종 질문(nl.json 패턴 준수)",
  "tag": "적절한 태그(유효한 태그 목록에서만 선택)"
}}
""".format(
        json.dumps(schema_info, ensure_ascii=False, indent=2),
        json.dumps(patterns, ensure_ascii=False, indent=2),
        question.original_question,
        question.final_question,
        question.tag,
        feedback,
        json.dumps(valid_tags, ensure_ascii=False)
    )
    
    response = await get_ollama_completion(improvement_prompt)
    
    try:
        # 정규식으로 JSON 객체 추출
        import re
        json_pattern = r'\{\s*".*"\s*:.*\}'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                improved = json.loads(json_str)
                # 원본 질문은 유지
                improved["original_question"] = question.original_question
                
                # 태그 검증
                if improved.get("tag") not in valid_tags:
                    improved["tag"] = question.tag if question.tag in valid_tags else valid_tags[0]
                    
                return EnhancedQuestion(**improved)
            except json.JSONDecodeError:
                pass
        
        # 정규식으로 찾지 못한 경우 다른 방법 시도
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            try:
                improved = json.loads(json_str)
                # 원본 질문은 유지
                improved["original_question"] = question.original_question
                
                # 태그 검증
                if improved.get("tag") not in valid_tags:
                    improved["tag"] = question.tag if question.tag in valid_tags else valid_tags[0]
                    
                return EnhancedQuestion(**improved)
            except json.JSONDecodeError:
                pass
                
        # 기본적인 개선 시도
        tag = question.tag if question.tag in valid_tags else valid_tags[0]
        return EnhancedQuestion(
            original_question=question.original_question,
            final_question=question.final_question,
            tag=tag
        )
    except Exception as e:
        # 모든 예외 처리
        print(f"질문 개선 중 오류 발생: {str(e)}")
        # 에러 발생 시 유효한 태그로 설정
        tag = question.tag if question.tag in valid_tags else valid_tags[0]
        return EnhancedQuestion(
            original_question=question.original_question,
            final_question=question.final_question,
            tag=tag
        )

@app.post("/table_info", response_model=Dict[str, Any])
async def get_table_info(request: TableInfo):
    """테이블의 스키마 정보와 유니크 값을 가져옴"""
    schema = get_table_schema(request.table_name)
    
    # 각 컬럼의 유니크 값 가져오기
    for column in schema["columns"]:
        column["unique_values"] = get_unique_values(
            request.table_name, 
            column["column_name"], 
            column["data_type"], 
            request.limit_unique_values
        )
    
    return schema

async def evaluate_and_improve_questions(questions: List[EnhancedQuestion], schema: Dict, valid_tags: List[str], nl_examples: List[Dict]) -> List[EnhancedQuestion]:
    """질문을 평가하고 nl.json 패턴에 맞게 개선"""
    
    # 간소화된 스키마 정보
    schema_for_evaluation = {
        "table_name": schema["table_name"],
        "columns": [{"column_name": col["column_name"], "comment": col.get("comment", "")} for col in schema["columns"]],
    }
    
    # nl.json 질문 패턴 추출
    patterns = []
    for example in nl_examples:
        if "original_question" in example and "final_question" in example:
            patterns.append({
                "original": example["original_question"],
                "final": example["final_question"]
            })
    
    evaluation_prompt = """
당신은 데이터 분석 질문 평가 및 개선 전문가입니다. 다음 질문 목록을 평가하고 각 질문을 nl.json 패턴에 맞게 개선해주세요.

테이블 스키마 정보:
{}

nl.json 질문 패턴 예시:
{}

현재 질문 목록:
{}

각 질문이 nl.json 패턴을 정확히 따르고 있는지 확인하고, 필요한 경우 다음과 같이 개선해주세요:
1. original_question은 유지하되, final_question이 nl.json 패턴을 따르도록 수정
2. 주차 표기 방식: "2023년 52W" -> "2023년 52주차에서"
3. 국가명 표기 방식: "브라질" -> "Brazil 국가의"
4. 문장 종결 방식: "~은 얼마인가?" -> "~을 알려줘." 또는 "~을 요약해줘."
5. 태그가 유효하지 않은 경우 적절한 태그로 수정 (유효한 태그: {})

각 질문에 대해 JSON 배열 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요.
[
  {{
    "original_question": "원래 질문(변경 없음)",
    "final_question": "개선된 최종 질문(nl.json 패턴 준수)",
    "tag": "적절한 태그",
    "score": 0.95
  }},
  ...
]
""".format(
        json.dumps(schema_for_evaluation, ensure_ascii=False, indent=2),
        json.dumps(patterns, ensure_ascii=False, indent=2),
        json.dumps([{"original_question": q.original_question, "final_question": q.final_question, "tag": q.tag} for q in questions], ensure_ascii=False, indent=2),
        json.dumps(valid_tags, ensure_ascii=False)
    )
    
    response = await get_ollama_completion(evaluation_prompt)
    
    try:
        # 텍스트에서 JSON 배열 추출하기
        import re
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                improved_questions = json.loads(json_str)
                # JSON 형식의 개선된 질문 목록으로 변환
                result = []
                for q in improved_questions:
                    if "original_question" in q and "final_question" in q and "tag" in q:
                        # 태그 유효성 검증
                        if q["tag"] not in valid_tags:
                            q["tag"] = valid_tags[0] if valid_tags else "getMainIndexState"
                        
                        # EnhancedQuestion 객체 생성
                        enhanced_q = EnhancedQuestion(
                            original_question=q["original_question"],
                            final_question=q["final_question"],
                            tag=q["tag"],
                            score=q.get("score", 0.8)
                        )
                        result.append(enhanced_q)
                
                return result
            except json.JSONDecodeError:
                pass
        
        # 정규식으로 찾지 못한 경우 원본 질문 반환
        return questions
    except Exception as e:
        # 모든 예외 처리
        print(f"질문 평가 및 개선 중 오류 발생: {str(e)}")
        return questions



@app.post("/generate_questions", response_model=QuestionGenResponse)
async def generate_questions(request: QuestionGenRequest):
    """예상 질문 목록 생성 - 개선된 버전"""
    # 테이블 스키마와 유니크 값 가져오기
    schema = get_table_schema(request.table_name)
    
    # 각 컬럼의 유니크 값 가져오기 (숫자 제외)
    for column in schema["columns"]:
        column["unique_values"] = get_unique_values(
            request.table_name, 
            column["column_name"], 
            column["data_type"], 
            10  # 각 컬럼당 최대 10개의 유니크 값 가져오기
        )
    
    # 컬럼 이름과 코멘트 매핑 작성
    column_mappings = []
    for col in schema["columns"]:
        mapping = {
            "column_name": col["column_name"],
            "comment": col.get("comment", "")
        }
        if mapping["comment"]:
            column_mappings.append(mapping)
    
    # nl.json 파일에서 예시 질문과 유효한 태그 로드
    nl_examples = load_nl_json()
    valid_tags = get_valid_tags()
    
    # 예시 질문과 패턴 추출 - 전체 nl.json을 사용
    question_patterns = []
    for example in nl_examples:
        if "original_question" in example and "final_question" in example:
            pattern = {
                "original": example["original_question"],
                "final": example["final_question"],
                "tag": example.get("tag", valid_tags[0] if valid_tags else "getMainIndexState")
            }
            question_patterns.append(pattern)
    
    # gemma3:27b 모델에 요청할 프롬프트 생성
    prompt = f"""
당신은 마케팅 전문가입니다. PostgreSQL 데이터베이스의 '{request.table_name}' 테이블에 대해 질문할 수 있는 예상 질문 목록을 생성해주세요.

## 테이블 정보
테이블명: {request.table_name}
행 수: {schema['row_count']}

## 컬럼 정보:
{json.dumps([{
    "column_name": col["column_name"],
    "data_type": col["data_type"],
    "is_nullable": col["is_nullable"],
    "comment": col.get("comment", ""),
    "unique_values_sample": col["unique_values"][:5] if col["unique_values"] else []
} for col in schema["columns"]], indent=2, ensure_ascii=False)}

## 기본 키:
{schema["primary_keys"]}

## 컬럼 명칭과 설명 (코멘트) 매핑:
{json.dumps(column_mappings, indent=2, ensure_ascii=False)}

## 사용자 질문 예시:
{json.dumps(request.example_questions, indent=2, ensure_ascii=False)}

## nl.json의 질문 패턴 예시 (아래 패턴을 엄격히 따라야 함):
{json.dumps(question_patterns, indent=2, ensure_ascii=False)}

## 유효한 태그 목록 (반드시 이 목록에서만 태그를 선택해야 함):
{json.dumps(valid_tags, indent=2, ensure_ascii=False)}

위 정보를 바탕으로 사용자가 이 테이블에 대해 할 수 있는 질문 {request.num_questions}개를 생성해주세요.

중요 지침:
1. **매우 중요**: original_question과 final_question의 형식은 반드시 nl.json의 예시 패턴을 따라야 합니다.
   - 예를 들어, nl.json에 "2024년46W 브라질 주요 지표 말해봐."가 있고 final_question이 "2024년46주차에서 Brazil 국가의 주요 지표 현황을 요약해줘."라면,
   - 새로 생성하는 질문도 "2023년 52주차 브라질 법인의 매출액은 얼마인가?"와 같은 original_question에 대해
   - final_question은 "2023년 52주차에서 Brazil 국가의 매출액 현황을 알려줘."와 같이 패턴을 엄격히 따라야 합니다.
   
2. 다음 패턴에 주의하세요:
   - 주차 표기: "2023년 52W" -> "2023년 52주차에서"
   - 국가명 표기: "브라질" -> "Brazil 국가의"
   - 문장 종결: "~은 얼마인가?" -> "~을 알려줘." 또는 "~을 요약해줘."

3. 데이터 분석 및 통계 관련 질문을 포함하세요.
4. 중복되지 않는 다양한 질문을 만들어주세요.
5. 각 컬럼의 유니크한 값을 활용하여 구체적인 질문을 만들어주세요.
6. 한글로 자연스러운 채팅형 질문을 생성해주세요.
7. 컬럼들을 통해 새로운 파라미터를 구해서 사용해도 좋습니다, 예를들어 CTC는 Cost per click을 의미합니다.
8. 질문이 너무 길어지지 않도록 주의해주세요.
9. AOV, CTR, CPC, CVR, ROAS 등의 지표 등을 활용한 질문들도 포함해주세요.
10. **중요**: 컬럼명 대신 해당 컬럼의 코멘트(설명)를 활용하여 더 자연스러운 질문을 생성해주세요.
11. **반드시** 유효한 태그 목록에서만 태그를 선택하세요. 다른 태그는 사용하지 마세요.

각 질문마다 아래 형식으로 JSON 객체를 만들어주세요:
{{
  "original_question": "사용자가 입력할만한 형태의 질문",
  "final_question": "정제되고 명확한 형태의 질문 (nl.json의 패턴을 엄격히 따름)", 
  "tag": "이 질문의 의도나 카테고리를 나타내는 태그(유효한 태그 목록에서만 선택)"
}}

응답은 다음과 같이 JSON 형식의 배열로 제공해주세요:
[
  {{
    "original_question": "...",
    "final_question": "...",
    "tag": "..."
  }},
  ...
]
"""
    
    # Ollama 모델에 질문 생성 요청
    response = await get_ollama_completion(prompt)
    
    try:
        # 정규식으로 JSON 배열 추출
        import re
        json_pattern = r'\[\s*\{.*\}\s*\]'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                questions_data = json.loads(json_str)
            except json.JSONDecodeError:
                # 정규식으로 찾았지만 파싱 실패한 경우
                raise ValueError("JSON 파싱 실패")
        else:
            # 정규식으로 찾지 못한 경우 다른 방법 시도
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                questions_data = json.loads(json_str)
            else:
                # JSON이 명확하게 감지되지 않으면 전체 문자열 파싱 시도
                questions_data = json.loads(response)
        
        # JSON 구조 검증 및 변환
        enhanced_questions = []
        for q in questions_data[:request.num_questions]:
            if isinstance(q, dict) and "original_question" in q and "final_question" in q and "tag" in q:
                # 태그 검증 - 유효한 태그가 아니면 기본 태그로 변경
                if q["tag"] not in valid_tags:
                    q["tag"] = valid_tags[0] if valid_tags else "getMainIndexState"
                enhanced_questions.append(EnhancedQuestion(**q))
            
        # 충분한 질문이 생성되지 않았으면 에러
        if len(enhanced_questions) < 1:
            raise ValueError("유효한 질문이 생성되지 않았습니다.")
            
    except Exception as e:
        # 모든 예외 처리 (JSON 파싱 실패 등)
        print(f"질문 생성 중 오류 발생: {str(e)}")
        
        # 응답에서 줄 단위로 텍스트 추출하여 기본 질문 생성
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        enhanced_questions = []
        
        for i, line in enumerate(lines[:request.num_questions]):
            # 기본 질문 형식으로 변환 (기본 태그 사용)
            default_tag = valid_tags[0] if valid_tags else "getMainIndexState"
            enhanced_questions.append(EnhancedQuestion(
                original_question=line,
                final_question=line,
                tag=default_tag
            ))
    
    # 생성된 질문 평가 및 개선
    enhanced_questions = await evaluate_and_improve_questions(enhanced_questions, schema, valid_tags, nl_examples)
    
    return {
        "generated_questions": enhanced_questions,
        "table_schema": schema
    }

@app.get("/valid_tags", response_model=List[str])
async def get_available_tags():
    """현재 시스템에서 사용 가능한 유효한 태그 목록을 반환"""
    return get_valid_tags()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
