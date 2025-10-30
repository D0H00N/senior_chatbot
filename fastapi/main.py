from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pymysql, os
from datetime import date, datetime, timedelta

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚙️ MySQL 연결
def get_mysql_connection():
    return pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="1224",
        database="senior_chatbot",
        cursorclass=pymysql.cursors.DictCursor
    )

# 🏠 홈: 알림요약 + 기본 목록
@app.get("/")
async def home(request: Request, q: str | None = Query(default=None)):
    conn = get_mysql_connection()
    cur = conn.cursor()

    # ① 오늘/최근 업로드 영상 기반 '알림형 요약' (최근 7일)
    cur.execute("""
        SELECT d.video_id, d.title, d.upload_date
        FROM chatbot_data AS d
        INNER JOIN chatbot_summary AS s
           ON s.video_id = d.video_id
        WHERE d.upload_date >= CURDATE() - INTERVAL 7 DAY
        GROUP BY d.video_id, d.title, d.upload_date
        ORDER BY d.upload_date DESC
        LIMIT 8
    """)
    recent = cur.fetchall()

    # ② 목록/검색 (제목 기준)
    if q:
        cur.execute("""
            SELECT d.video_id, d.title, MAX(d.upload_date) AS upload_date
            FROM chatbot_data AS d
            INNER JOIN chatbot_summary AS s
               ON s.video_id = d.video_id
            WHERE d.title LIKE %s
            GROUP BY d.video_id, d.title
            ORDER BY upload_date DESC
            LIMIT 24
        """, (f"%{q}%",))
    else:
        cur.execute("""
            SELECT d.video_id, d.title, MAX(d.upload_date) AS upload_date
            FROM chatbot_data AS d
            INNER JOIN chatbot_summary AS s
               ON s.video_id = d.video_id
            GROUP BY d.video_id, d.title
            ORDER BY upload_date DESC
            LIMIT 24
        """)
    videos = cur.fetchall()

    conn.close()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "recent": recent, "videos": videos, "query": q or ""}
    )

# ▶️ 특정 video_id의 요약 1개 반환
@app.get("/api/summary/{video_id}")
async def get_summary(video_id: str):
    print(f"[HIT] /api/summary/{video_id}")
    conn = get_mysql_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT summary_text, video_url
        FROM chatbot_summary
        WHERE video_id=%s
    """, (video_id,))
    row = cur.fetchone(); conn.close()
    if not row:
        return JSONResponse({"error":"summary not found"}, status_code=404)
    return { "summary_text": row["summary_text"], "video_url": row["video_url"]}


# 🔍 제목 검색 API (ajax 사용 시)
@app.get("/api/search")
async def api_search(q: str):
    conn = get_mysql_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT d.video_id, d.title, MAX(d.upload_date) AS upload_date
        FROM chatbot_data AS d
        INNER JOIN chatbot_summary AS s
           ON s.video_id = d.video_id
        WHERE title LIKE %s
        GROUP BY d.video_id, d.title
        ORDER BY upload_date DESC
        LIMIT 24
    """, (f"%{q}%",))
    rows = cur.fetchall()
    conn.close()
    return JSONResponse(rows)