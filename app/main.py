import os, re, json, tempfile
from datetime import datetime, timedelta
from typing import List, Optional

import httpx
import yt_dlp
import whisper
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# ─── Config ──────────────────────────────────────────────────────────────────
DATABASE_URL        = os.getenv("DATABASE_URL", "")
SECRET_KEY          = os.getenv("SECRET_KEY", "changeme-please")
DEEPSEEK_API_KEY    = os.getenv("DEEPSEEK_API_KEY", "")
ADMIN_USERNAME      = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD      = os.getenv("ADMIN_PASSWORD", "admin123")
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "base")
TOKEN_EXPIRE_MIN    = 60 * 24 * 7

# ─── Database ─────────────────────────────────────────────────────────────────
engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

def get_db():
    db = SessionLocal()
    try:    yield db
    finally: db.close()

# ─── Models ───────────────────────────────────────────────────────────────────
class Category(Base):
    __tablename__ = "categories"
    id         = Column(Integer, primary_key=True)
    name       = Column(String(100), nullable=False)
    name_ar    = Column(String(100), nullable=False)
    icon       = Column(String(50))
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    courses    = relationship("Course", back_populates="category")

class Course(Base):
    __tablename__ = "courses"
    id              = Column(Integer, primary_key=True)
    title           = Column(String(200), nullable=False)
    title_ar        = Column(String(200))
    description     = Column(Text)
    description_ar  = Column(Text)
    thumbnail_url   = Column(String(500))
    category_id     = Column(Integer, ForeignKey("categories.id"))
    published_at    = Column(DateTime(timezone=True))
    is_published    = Column(Boolean, default=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    category        = relationship("Category", back_populates="courses")
    lectures        = relationship("Lecture", back_populates="course", order_by="Lecture.order")

class Lecture(Base):
    __tablename__ = "lectures"
    id               = Column(Integer, primary_key=True)
    course_id        = Column(Integer, ForeignKey("courses.id"))
    title            = Column(String(200), nullable=False)
    title_ar         = Column(String(200))
    youtube_url      = Column(String(500), nullable=False)
    youtube_id       = Column(String(50))
    thumbnail_url    = Column(String(500))
    order            = Column(Integer, default=0)
    duration_seconds = Column(Integer)
    status           = Column(String(30), default="pending")
    error_message    = Column(Text)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())
    course           = relationship("Course", back_populates="lectures")
    subtitles        = relationship("Subtitle", back_populates="lecture")
    segments         = relationship("Segment", back_populates="lecture")
    ratings          = relationship("Rating", back_populates="lecture")
    progresses       = relationship("UserProgress", back_populates="lecture")

class Subtitle(Base):
    __tablename__ = "subtitles"
    id         = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id"))
    start_time = Column(Float, nullable=False)
    end_time   = Column(Float, nullable=False)
    text_en    = Column(Text)
    text_ar    = Column(Text)
    lecture    = relationship("Lecture", back_populates="subtitles")

class Segment(Base):
    __tablename__ = "segments"
    id         = Column(Integer, primary_key=True)
    lecture_id = Column(Integer, ForeignKey("lectures.id"))
    title_ar   = Column(String(300))
    start_time = Column(Float)
    end_time   = Column(Float)
    order      = Column(Integer)
    lecture    = relationship("Lecture", back_populates="segments")

class Rating(Base):
    __tablename__ = "ratings"
    id          = Column(Integer, primary_key=True)
    lecture_id  = Column(Integer, ForeignKey("lectures.id"))
    device_id   = Column(String(100))
    is_positive = Column(Boolean)
    feedback    = Column(Text)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    lecture     = relationship("Lecture", back_populates="ratings")

class UserProgress(Base):
    __tablename__ = "user_progress"
    id            = Column(Integer, primary_key=True)
    lecture_id    = Column(Integer, ForeignKey("lectures.id"))
    device_id     = Column(String(100))
    last_position = Column(Float, default=0)
    is_completed  = Column(Boolean, default=False)
    updated_at    = Column(DateTime(timezone=True), server_default=func.now())
    lecture       = relationship("Lecture", back_populates="progresses")

# ─── Schemas ──────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class CategoryCreate(BaseModel):
    name: str
    name_ar: str
    icon: Optional[str] = None

class CategoryOut(BaseModel):
    id: int; name: str; name_ar: str; icon: Optional[str]; is_active: bool
    class Config: from_attributes = True

class LectureInput(BaseModel):
    youtube_url: str
    title: str

class CourseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    category_id: int
    published_at: Optional[datetime] = None
    lectures: List[LectureInput]

class SubtitleOut(BaseModel):
    start_time: float; end_time: float; text_ar: str
    class Config: from_attributes = True

class SegmentOut(BaseModel):
    title_ar: str; start_time: float; end_time: float; order: int
    class Config: from_attributes = True

class LectureOut(BaseModel):
    id: int; title: str; title_ar: Optional[str]; youtube_id: Optional[str]
    order: int; duration_seconds: Optional[int]; status: str
    subtitles: List[SubtitleOut] = []; segments: List[SegmentOut] = []; likes_count: int = 0
    class Config: from_attributes = True

class CourseOut(BaseModel):
    id: int; title: str; title_ar: Optional[str]; description: Optional[str]
    description_ar: Optional[str]; thumbnail_url: Optional[str]; category_id: int
    is_published: bool; published_at: Optional[datetime]; lectures: List[LectureOut] = []
    class Config: from_attributes = True

class CourseListItem(BaseModel):
    id: int; title: str; title_ar: Optional[str]; description_ar: Optional[str]
    thumbnail_url: Optional[str]; category_id: int; lectures_count: int = 0
    class Config: from_attributes = True

class RatingCreate(BaseModel):
    device_id: str; is_positive: bool; feedback: Optional[str] = None

class ProgressUpdate(BaseModel):
    device_id: str; last_position: float; is_completed: bool = False

# ─── Auth ─────────────────────────────────────────────────────────────────────
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/admin/login")

def create_access_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm="HS256")

def verify_admin(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403)
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── Pipeline ─────────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model

def extract_youtube_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|embed/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else ""

def download_audio(youtube_url: str, output_dir: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    return os.path.join(output_dir, "audio.mp3"), info.get("duration", 0)

def transcribe_audio(audio_path: str):
    result = get_whisper_model().transcribe(audio_path, task="transcribe", verbose=False)
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]

async def call_deepseek(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "max_tokens": 4000, "temperature": 0.3}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def translate_subtitles(segments):
    translated = []
    for i in range(0, len(segments), 50):
        batch = segments[i:i+50]
        numbered = "\n".join([f"{j+1}. {s['text']}" for j, s in enumerate(batch)])
        response = await call_deepseek(f"ترجم من الإنجليزية للعربية الفصحى المبسطة، أعد الترقيم فقط:\n\n{numbered}")
        lines = response.strip().split("\n")
        for j, seg in enumerate(batch):
            line = re.sub(r"^\d+\.\s*", "", lines[j].strip() if j < len(lines) else "")
            translated.append({"start": seg["start"], "end": seg["end"], "text_en": seg["text"], "text_ar": line})
    return translated

async def generate_segments(subtitles, lecture_title: str):
    full_text = " ".join([f"[{s['start']:.0f}s] {s['text_en']}" for s in subtitles[:200]])
    response = await call_deepseek(
        f'المحاضرة: "{lecture_title}"\n{full_text[:6000]}\n\nقسّم لفقرات منطقية (3-8)، أعد JSON فقط:\n[{{"title_ar":"...","start_time":0}}]'
    )
    try:
        m = re.search(r"\[.*\]", response, re.DOTALL)
        if m:
            raw = json.loads(m.group())
            result = []
            for idx, seg in enumerate(raw):
                start = float(seg.get("start_time", 0))
                end = float(raw[idx+1]["start_time"]) if idx+1 < len(raw) else (subtitles[-1]["end"] if subtitles else start+60)
                result.append({"title_ar": seg.get("title_ar", f"الجزء {idx+1}"), "start_time": start, "end_time": end, "order": idx})
            return result
    except Exception as e:
        print(f"Segment error: {e}")
    return []

async def translate_metadata(title: str, description: str):
    response = await call_deepseek(f'ترجم للعربية، JSON فقط:\n{{"title_ar":"...","description_ar":"..."}}\n\nالعنوان: {title}\nالوصف: {description or ""}')
    try:
        m = re.search(r"\{.*\}", response, re.DOTALL)
        if m: return json.loads(m.group())
    except: pass
    return {"title_ar": title, "description_ar": description}

async def process_lecture(lecture_id: int, youtube_url: str, title: str):
    db = SessionLocal()
    try:
        lec = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        lec.status = "processing"; db.commit()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path, duration = download_audio(youtube_url, tmpdir)
            lec.duration_seconds = int(duration); db.commit()
            raw = transcribe_audio(audio_path)
            translated = await translate_subtitles(raw)
            for s in translated:
                db.add(Subtitle(lecture_id=lecture_id, start_time=s["start"], end_time=s["end"], text_en=s["text_en"], text_ar=s["text_ar"]))
            for seg in await generate_segments(translated, title):
                db.add(Segment(lecture_id=lecture_id, title_ar=seg["title_ar"], start_time=seg["start_time"], end_time=seg["end_time"], order=seg["order"]))
            lec.status = "done"; db.commit()
    except Exception as e:
        lec = db.query(Lecture).filter(Lecture.id == lecture_id).first()
        lec.status = "failed"; lec.error_message = str(e); db.commit()
        print(f"Pipeline error [{lecture_id}]: {e}")
    finally:
        db.close()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Craft Courses API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Category).count() == 0:
            db.add_all([
                Category(name="Sewing & Embroidery",  name_ar="الخياطة والتطريز",    icon="🧵"),
                Category(name="Woodworking",           name_ar="النجارة وأعمال الخشب", icon="🪵"),
                Category(name="Drawing & Arts",        name_ar="الرسم والفنون",        icon="🎨"),
                Category(name="Candles & Soap",        name_ar="الشموع والصابون",      icon="🕯️"),
                Category(name="Home Maintenance",      name_ar="الصيانة المنزلية",     icon="🔧"),
            ])
            db.commit()
    finally:
        db.close()

@app.get("/health")
def health(): return {"status": "ok"}

# ── Admin ────────────────────────────────────────────────────────────────────
@app.post("/api/admin/login", response_model=TokenResponse)
def login(req: LoginRequest):
    if req.username != ADMIN_USERNAME or req.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": req.username, "role": "admin"})}

@app.get("/api/admin/categories", response_model=List[CategoryOut])
def admin_get_categories(db: Session = Depends(get_db), _=Depends(verify_admin)):
    return db.query(Category).all()

@app.post("/api/admin/categories", response_model=CategoryOut)
def admin_create_category(data: CategoryCreate, db: Session = Depends(get_db), _=Depends(verify_admin)):
    cat = Category(**data.model_dump()); db.add(cat); db.commit(); db.refresh(cat); return cat

@app.delete("/api/admin/categories/{cat_id}")
def admin_delete_category(cat_id: int, db: Session = Depends(get_db), _=Depends(verify_admin)):
    cat = db.query(Category).filter(Category.id == cat_id).first()
    if not cat: raise HTTPException(404)
    cat.is_active = False; db.commit(); return {"ok": True}

@app.get("/api/admin/courses", response_model=List[CourseListItem])
def admin_list_courses(db: Session = Depends(get_db), _=Depends(verify_admin)):
    return [CourseListItem(id=c.id, title=c.title, title_ar=c.title_ar, description_ar=c.description_ar,
            thumbnail_url=c.thumbnail_url, category_id=c.category_id, lectures_count=len(c.lectures))
            for c in db.query(Course).all()]

@app.post("/api/admin/courses")
async def admin_create_course(data: CourseCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db), _=Depends(verify_admin)):
    meta = await translate_metadata(data.title, data.description or "")
    course = Course(title=data.title, title_ar=meta.get("title_ar"), description=data.description,
                    description_ar=meta.get("description_ar"), category_id=data.category_id,
                    published_at=data.published_at, is_published=True)
    db.add(course); db.commit(); db.refresh(course)
    for idx, lec_input in enumerate(data.lectures):
        yt_id = extract_youtube_id(lec_input.youtube_url)
        lec = Lecture(course_id=course.id, title=lec_input.title, youtube_url=lec_input.youtube_url,
                      youtube_id=yt_id, order=idx, status="pending",
                      thumbnail_url=f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg" if yt_id else None)
        db.add(lec); db.commit(); db.refresh(lec)
        background_tasks.add_task(process_lecture, lec.id, lec_input.youtube_url, lec_input.title)
    return {"course_id": course.id, "lectures_queued": len(data.lectures)}

@app.delete("/api/admin/courses/{course_id}")
def admin_delete_course(course_id: int, db: Session = Depends(get_db), _=Depends(verify_admin)):
    c = db.query(Course).filter(Course.id == course_id).first()
    if not c: raise HTTPException(404)
    c.is_published = False; db.commit(); return {"ok": True}

@app.get("/api/admin/courses/{course_id}/status")
def admin_course_status(course_id: int, db: Session = Depends(get_db), _=Depends(verify_admin)):
    return [{"id": l.id, "title": l.title, "status": l.status, "error": l.error_message}
            for l in db.query(Lecture).filter(Lecture.course_id == course_id).all()]

# ── Public ───────────────────────────────────────────────────────────────────
@app.get("/api/categories", response_model=List[CategoryOut])
def get_categories(db: Session = Depends(get_db)):
    return db.query(Category).filter(Category.is_active == True).all()

@app.get("/api/courses", response_model=List[CourseListItem])
def list_courses(category_id: int = None, db: Session = Depends(get_db)):
    q = db.query(Course).filter(Course.is_published == True)
    if category_id: q = q.filter(Course.category_id == category_id)
    return [CourseListItem(id=c.id, title=c.title, title_ar=c.title_ar, description_ar=c.description_ar,
            thumbnail_url=c.thumbnail_url, category_id=c.category_id, lectures_count=len(c.lectures))
            for c in q.all()]

@app.get("/api/courses/{course_id}", response_model=CourseOut)
def get_course(course_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id, Course.is_published == True).first()
    if not course: raise HTTPException(404)
    lectures_out = []
    for lec in course.lectures:
        likes = db.query(func.count(Rating.id)).filter(Rating.lecture_id == lec.id, Rating.is_positive == True).scalar()
        lectures_out.append(LectureOut(
            id=lec.id, title=lec.title, title_ar=lec.title_ar, youtube_id=lec.youtube_id,
            order=lec.order, duration_seconds=lec.duration_seconds, status=lec.status,
            subtitles=[SubtitleOut(start_time=s.start_time, end_time=s.end_time, text_ar=s.text_ar)
                       for s in sorted(lec.subtitles, key=lambda x: x.start_time)],
            segments=[SegmentOut(title_ar=s.title_ar, start_time=s.start_time, end_time=s.end_time, order=s.order)
                      for s in sorted(lec.segments, key=lambda x: x.order)],
            likes_count=likes))
    return CourseOut(id=course.id, title=course.title, title_ar=course.title_ar, description=course.description,
                     description_ar=course.description_ar, thumbnail_url=course.thumbnail_url,
                     category_id=course.category_id, is_published=course.is_published,
                     published_at=course.published_at, lectures=lectures_out)

@app.post("/api/lectures/{lecture_id}/rate")
def rate_lecture(lecture_id: int, data: RatingCreate, db: Session = Depends(get_db)):
    existing = db.query(Rating).filter(Rating.lecture_id == lecture_id, Rating.device_id == data.device_id).first()
    if existing: existing.is_positive = data.is_positive; existing.feedback = data.feedback
    else: db.add(Rating(lecture_id=lecture_id, device_id=data.device_id, is_positive=data.is_positive, feedback=data.feedback))
    db.commit(); return {"ok": True}

@app.post("/api/lectures/{lecture_id}/progress")
def update_progress(lecture_id: int, data: ProgressUpdate, db: Session = Depends(get_db)):
    p = db.query(UserProgress).filter(UserProgress.lecture_id == lecture_id, UserProgress.device_id == data.device_id).first()
    if p: p.last_position = data.last_position; p.is_completed = data.is_completed
    else: db.add(UserProgress(lecture_id=lecture_id, device_id=data.device_id, last_position=data.last_position, is_completed=data.is_completed))
    db.commit(); return {"ok": True}

@app.get("/api/progress/{device_id}")
def get_progress(device_id: str, db: Session = Depends(get_db)):
    return [{"lecture_id": p.lecture_id, "last_position": p.last_position, "is_completed": p.is_completed}
            for p in db.query(UserProgress).filter(UserProgress.device_id == device_id).all()]
