# ------------------------------------------------------------
# YouTube 채널 → 오디오/비디오 확보 → Whisper 전사/청크 →
# 청크별 중간 프레임 이미지 캡처 → 임베딩 → Spark Parquet 저장
# (불용어가 '부분'으로라도 포함되면 해당 청크 전체 DROP)
# ------------------------------------------------------------

import json
import math
import time
import re
from pathlib import Path
from datetime import datetime

from yt_dlp import YoutubeDL
from pytube import YouTube  # fallback
import whisper, torch, ffmpeg
from sentence_transformers import SentenceTransformer

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

# ───────────────────────── 설정 ─────────────────────────
CHANNEL_VIDEOS_URL = "https://www.youtube.com/@100junsa/videos"

# 작업 디렉토리
WORK_DIR  = Path("audio");        WORK_DIR.mkdir(exist_ok=True)
TRANS_DIR = Path("transcripts");  TRANS_DIR.mkdir(exist_ok=True)
SEG_DIR   = Path("segments");     SEG_DIR.mkdir(exist_ok=True)
IMG_DIR   = Path("chunk_images"); IMG_DIR.mkdir(exist_ok=True)

# 장비
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델
whisper_model = whisper.load_model("base", device=device)
embed_model   = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# yt_dlp 옵션
YDL_LIST_OPTS = {
    "extract_flat": True,  # 리스트에서 빠른 메타 수집
    "quiet": False,
    "no_warnings": True,
    "ignoreerrors": True
}
YDL_SINGLE_OPTS = {
    "quiet": False,
    "no_warnings": True,
    "ignoreerrors": True
}

# ───────────────────────── 불용어 (부분 매칭이면 DROP) ─────────────────────────
STOPWORD_PHRASES = [
    # 인사/구독/감사/채널 멘트 (필요에 맞게 추가/수정하세요)
    "좋아요와 구독 눌러주세요", "좋아요와 구독", "구독과 좋아요", "구독 눌러", "좋아요 부탁",
    "알림 설정", "알림설정", "subscribe", "like","영상링크도","영상링크","함께","공유해주세요"
    "안녕하세요", "안녕하십니까", "오늘도", "영상", "100세", "시대를", "준비하는", "사람들","100세 시대를","오늘도 끝까지 시청"
    "시청해주셔서 감사합니다", "시청 감사합니다", "끝까지 시청", "좋아요도", "좋아요","영상","보시기 전에","구독","구독 눌러주시구요","눌러주시구요"
    "백세시대를 준비하는 사람들", "채널","오늘","영상이","유용했다면","오늘영상", "오늘 영상", "시작합니다","백세시대를","백세시대"
]
# 원문/정규화(공백·특수문자 제거, 소문자) 두 레이어 검사
def _normalize_simple(s: str) -> str:
    s = s or ""
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s.lower()

_STOP_PATTERNS_RAW  = [re.compile(re.escape(p), re.IGNORECASE) for p in STOPWORD_PHRASES]
_STOP_PATTERNS_NORM = [re.compile(re.escape(_normalize_simple(p))) for p in STOPWORD_PHRASES]

def contains_stopphrase(text: str) -> bool:
    """텍스트에 불용어가 '부분'으로라도 포함되면 True (DROP)"""
    if not text:
        return False
    for pat in _STOP_PATTERNS_RAW:
        if pat.search(text):
            return True
    norm = _normalize_simple(text)
    for pat in _STOP_PATTERNS_NORM:
        if pat.search(norm):
            return True
    return False

# ───────────────────────── 유틸 ─────────────────────────
def parse_date_yyyymmdd(s: str):
    """YYYYMMDD → YYYY-MM-DD (파싱 실패시 None)"""
    try:
        return datetime.strptime(s, "%Y%m%d").strftime("%Y-%m-%d")
    except Exception:
        return None

def fetch_meta_single(video_id: str):
    """단일 영상 메타를 yt_dlp로 재조회(정확한 upload_date/제목 확보)"""
    with YoutubeDL(YDL_SINGLE_OPTS) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    title = (info or {}).get("title") or ""
    up = (info or {}).get("upload_date")  # YYYYMMDD
    upload_date = parse_date_yyyymmdd(up) if up else None
    return title, upload_date

def fetch_meta_fallback_pytube(video_id: str):
    """pytube fallback: publish_date → YYYY-MM-DD"""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        if yt.publish_date:
            return yt.title or "", yt.publish_date.strftime("%Y-%m-%d")
    except Exception:
        pass
    return "", None

def ensure_wav(vid: str) -> Path:
    """오디오(webm) 다운로드 후 16kHz mono wav로 변환"""
    webm = WORK_DIR / f"{vid}.webm"
    wav  = WORK_DIR / f"{vid}.wav"
    if not webm.exists():
        with YoutubeDL({
            "format": "bestaudio",
            "outtmpl": str(webm),
            "quiet": True, "no_warnings": True
        }) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    if not wav.exists():
        ffmpeg.input(str(webm)).output(str(wav), ac=1, ar="16000", format="wav")\
              .overwrite_output().run(quiet=True)
    return wav

def ensure_video_mp4(vid: str) -> Path:
    """
    썸네일(프레임) 추출용 비디오 확보.
    원본을 mkv로 내려받고 mp4로 remux(가능시 copy).
    """
    mp4 = WORK_DIR / f"{vid}.mp4"
    if mp4.exists():
        return mp4

    tmp_media = WORK_DIR / f"{vid}.mkv"
    if not tmp_media.exists():
        with YoutubeDL({
            "format": "bv*+ba/best",
            "outtmpl": str(tmp_media),
            "merge_output_format": "mkv",
            "quiet": True, "no_warnings": True
        }) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])

    try:
        (
            ffmpeg
            .input(str(tmp_media))
            .output(str(mp4), c="copy", movflags="+faststart")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        # 코덱/컨테이너 이슈 시 재인코딩
        (
            ffmpeg
            .input(str(tmp_media))
            .output(str(mp4), vcodec="libx264", acodec="aac", movflags="+faststart")
            .overwrite_output()
            .run(quiet=True)
        )
    return mp4

def snapshot_frame(video_path: Path, ts_sec: float, out_path: Path):
    """ts 지점에서 한 프레임을 이미지로 저장 (빠른 탐색: input(ss=...))"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(video_path), ss=ts_sec)
        .output(str(out_path), vframes=1, format="image2", **{"qscale:v": 2})
        .overwrite_output()
        .run(quiet=True)
    )

def load_or_transcribe(vid: str, wav_path: Path):
    """로컬에 segments/text 있으면 로드, 없으면 Whisper로 전사"""
    txt = TRANS_DIR / f"{vid}.txt"
    seg = SEG_DIR / f"{vid}_segments.json"
    if txt.exists() and seg.exists():
        return json.loads(seg.read_text(encoding="utf-8"))

    print(f"[TRANS] {vid}: running Whisper")
    result = whisper_model.transcribe(str(wav_path), fp16=torch.cuda.is_available())
    segments = result["segments"]
    seg.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    txt.write_text(result["text"].strip(), encoding="utf-8")
    print(f"[OK]    {vid}: {len(segments)} chunks saved")
    return segments

def safe_midpoint(start_s: float, end_s: float) -> float:
    """중간 지점 타임스탬프(안전 여유 0.2초)"""
    dur = max(0.0, end_s - start_s)
    return max(0.2, start_s + max(0.2, dur / 2.0))

# ─────────────────────── 1) 채널 영상 목록 수집 ───────────────────────
with YoutubeDL(YDL_LIST_OPTS) as ydl:
    info = ydl.extract_info(CHANNEL_VIDEOS_URL, download=False)

video_meta = []
for e in info.get("entries", []):
    if not e or not e.get("id"):
        continue
    vid = e["id"]
    title = e.get("title", "") or ""
    up_raw = e.get("upload_date")  # 종종 없음
    upload_date = parse_date_yyyymmdd(up_raw) if up_raw else None

    # 누락 시 단건 재조회 → 그래도 없으면 pytube fallback
    if not upload_date or upload_date == "2000-01-01":
        t2, d2 = fetch_meta_single(vid)
        if d2:
            title = title or t2
            upload_date = d2
        else:
            t3, d3 = fetch_meta_fallback_pytube(vid)
            if d3:
                title = title or t3
                upload_date = d3
        time.sleep(0.2)

    video_meta.append({
        "id": vid,
        "title": title,
        "upload_date": upload_date  # YYYY-MM-DD or None
    })

print(f"가져온 영상 수: {len(video_meta)}")

# ─────────────────────── 2) 전사/청크 + 이미지 + 임베딩 ────────────────
all_rows = []
for meta in video_meta:
    vid = meta["id"]
    title = meta.get("title", "") or ""
    upload_date = meta.get("upload_date") or None

    try:
        # 오디오 & 전사
        wav_path = ensure_wav(vid)
        segments = load_or_transcribe(vid, wav_path)

        # 프레임 캡처용 비디오
        mp4_path = ensure_video_mp4(vid)

        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            # ── 불용어가 '부분'으로라도 포함되면 통째로 버림 ──
            if contains_stopphrase(text):
                continue

            start_s = float(seg["start"])
            end_s   = float(seg["end"])
            mid_s   = safe_midpoint(start_s, end_s)

            # 이미지 파일 경로 (예: chunk_images/<vid>/<chunk_id>.jpg)
            img_out = IMG_DIR / vid / f'{seg["id"]}.jpg'
            if not img_out.exists():
                try:
                    snapshot_frame(mp4_path, mid_s, img_out)
                except Exception as ie:
                    print(f"[WARN] snapshot failed {vid}-{seg['id']}: {ie}")

            # 임베딩
            embedding = embed_model.encode(text).tolist()

            all_rows.append({
                "video_id":    vid,
                "title":       title,
                "upload_date": upload_date,
                "chunk_id":    int(seg["id"]),
                "start_time":  str(datetime.utcfromtimestamp(start_s).time()),
                "end_time":    str(datetime.utcfromtimestamp(end_s).time()),
                "text":        text,
                "embedding":   embedding,
                "image_path":  str(img_out.as_posix()) if img_out.exists() else None
            })

    except Exception as e:
        print(f"[ERROR] {vid}: {e}")

# ─────────────────────── 3) Spark 저장 ────────────────────────────────
spark = SparkSession.builder \
    .appName("TranscriptEmbeddingChunks") \
    .config("spark.pyspark.driver.python", "C:/project/senior_chatbot/venv/Scripts/python.exe") \
    .config("spark.pyspark.python",        "C:/project/senior_chatbot/venv/Scripts/python.exe") \
    .config("spark.sql.session.timeZone",  "Asia/Seoul") \
    .getOrCreate()

schema = StructType([
    StructField("video_id",    StringType(),  False),
    StructField("title",       StringType(),  True),
    StructField("upload_date", StringType(),  True),   # YYYY-MM-DD
    StructField("chunk_id",    IntegerType(), False),
    StructField("start_time",  StringType(),  True),
    StructField("end_time",    StringType(),  True),
    StructField("text",        StringType(),  True),
    StructField("embedding",   ArrayType(DoubleType()), True),
    StructField("image_path",  StringType(),  True),
])

df = spark.createDataFrame(all_rows, schema=schema)
df.write.mode("overwrite").parquet("file:///C:/project/senior_chatbot/df_text_embed")
print(" Parquet 저장 완료: file:///C:/project/senior_chatbot/df_text_embed")


