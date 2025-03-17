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

# 모델 정의
class TableInfo(BaseModel):
    table_name: str
    limit_unique_values: int = Field(10, description="각 컬럼당 가져올 유니크 값의 최대 개수")
    
class QuestionGenRequest(BaseModel):
    table_name: str
    example_questions: List[str] = Field(..., description="사용자 질문 예시 목록")
    num_questions: int = Field(10, description="생성할 질문 개수")

class QuestionGenResponse(BaseModel):
    generated_questions: List[str]
    table_schema: Dict[str, Any]
    
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

@app.post("/generate_questions", response_model=QuestionGenResponse)
async def generate_questions(request: QuestionGenRequest):
    """예상 질문 목록 생성"""
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
    
    # gemma3:27b 모델에 요청할 프롬프트 생성
    #당신은 데이터베이스 질문 생성 전문가입니다. 사용자가 PostgreSQL 데이터베이스의 '{request.table_name}' 테이블에 대해 질문할 수 있는 예상 질문 목록을 생성해주세요.
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

위 정보를 바탕으로 사용자가 이 테이블에 대해 할 수 있는 질문 {request.num_questions}개를 생성해주세요.
- 데이터 분석 및 통계 관련 질문을 포함하세요.
- 중복되지 않는 다양한 질문을 만들어주세요.
- 각 컬럼의 유니크한 값을 활용하여 구체적인 질문을 만들어주세요.
- 질문마다 번호를 붙이지 말고, 각 질문을 별도의 줄에 작성해 주세요.
- 한글로 자연스러운 채팅형 질문을 생성해주세요.
- **중요**: 컬럼명 대신 해당 컬럼의 코멘트(설명)를 활용하여 더 자연스러운 질문을 생성해주세요.
  예를 들어, coupon_nm이라는 컬럼의 코멘트가 쿠폰 명칭이라면, 
  "LATAM 지역에서 coupon_nm이 비어있는 거래의 비율은 얼마인가?" 대신
  "LATAM 지역에서 쿠폰 할인이 없는 거래의 비율은 얼마인가?"와 같이 작성해주세요.

질문 목록:
"""
    
    # Ollama 모델에 질문 생성 요청
    response = await get_ollama_completion(prompt)
    
    # 응답에서 질문 목록 추출
    questions = [q.strip().replace("'", "") for q in response.split('\n') if q.strip()]
    
    # 중복 제거 및 요청된 수에 맞게 조정
    unique_questions = list(dict.fromkeys(questions))
    generated_questions = unique_questions[:request.num_questions]
    
    return {
        "generated_questions": generated_questions,
        "table_schema": schema
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
