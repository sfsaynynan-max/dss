from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.api import admin, public
from app.models import models

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Learn Arabic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin.router)
app.include_router(public.router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup():
    from app.core.database import SessionLocal
    from app.models.models import Category
    db = SessionLocal()
    try:
        if db.query(Category).count() == 0:
            defaults = [
                Category(name="Sewing & Embroidery", name_ar="الخياطة والتطريز", icon="🧵"),
                Category(name="Woodworking", name_ar="النجارة وأعمال الخشب", icon="🪵"),
                Category(name="Drawing & Arts", name_ar="الرسم والفنون", icon="🎨"),
                Category(name="Candles & Soap", name_ar="الشموع والصابون", icon="🕯️"),
                Category(name="Home Maintenance", name_ar="الصيانة المنزلية", icon="🔧"),
            ]
            db.add_all(defaults)
            db.commit()
    finally:
        db.close()
