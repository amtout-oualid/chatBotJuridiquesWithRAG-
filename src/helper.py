from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def loadPdfFilles(path):
    # allow either relative (to workspace root) or absolute paths
    path_obj = Path(path)
    if not path_obj.is_absolute():
        # assume notebook is running from the research folder
        base = Path.cwd().parent if Path.cwd().name == "research" else Path.cwd()
        path_obj = base / path_obj

    loader=DirectoryLoader(
        str(path_obj),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents
# loadPdfFilles with relative path; function will resolve to project root
extractedData = loadPdfFilles("data")

from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import json
import os
import re  # <--- زدنا هادي باش نقيو النص

# ==========================================
# 1. BULLETPROOF PATHING
# ==========================================
current_path = Path.cwd()

# If running from inside the 'research' folder, go up one level
if current_path.name == "research":
    BASE_DIR = current_path.parent 
else:
    # Failsafe: Hardcoded absolute path
    BASE_DIR = Path(r"C:\Users\vivobook\Desktop\INPT\Me\Project\developpementProject2\chatBotJuridiquesWithRAG-")

PDF_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "data" / "pages"
OCR_OUT_DIR = BASE_DIR / "artifacts" / "ocr"

# ==========================================
# 2. CONFIGURATION
# ==========================================
DPI = 300
POPPLER_PATH = r"C:\poppler\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SLICE_HEIGHT = 240 

# sanity checks for external dependencies
if not Path(POPPLER_PATH).exists():
    print(f"⚠️  Poppler path {POPPLER_PATH} does not exist. PDF conversion may fail.")

if not Path(pytesseract.pytesseract.tesseract_cmd).exists():
    print(f"⚠️  Tesseract binary {pytesseract.pytesseract.tesseract_cmd} not found. OCR may fail.")

# ==========================================
# 3. TEXT CLEANING FUNCTION (هنا فين كاين السر)
# ==========================================
def clean_text(text):
    if not text:
        return ""
    # 1. تبديل الرجوع للسطر و Tab بفراغ عادي
    text = re.sub(r'[\n\t\r]+', ' ', text)
    # 2. مسح الرموز والإيموجيز (نخليو غير الحروف، الأرقام، والترقيم الأساسي)
    text = re.sub(r'[^\w\s.,;:!؟،؛"\'\(\)\-/]', '', text)
    # 3. مسح الفراغات الزايدة (باش مايبقاش ليسباس كبير بين الكلمات)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_patches(img, slice_height=SLICE_HEIGHT):
    w, h = img.size
    patches = []
    coords = []
    for top in range(0, h, slice_height):
        box = (0, top, w, min(top + slice_height, h))
        patch = img.crop(box)
        patches.append(patch)
        coords.append(box)
    return patches, coords

def ocr_patches(patches):
    texts = []
    total = len(patches)
    for i, patch in enumerate(patches):
        # Progress indicator so you know it's not frozen
        print(f"      -> Extracting text from patch {i+1}/{total}...", end="\r")
        text = pytesseract.image_to_string(patch, lang='ara', config='--psm 6')
        
        # نطبقو التنظيف على النص المخرج من الصورة
        cleaned = clean_text(text)
        texts.append(cleaned)
        
    print() # Move to the next line after finishing the patches
    return texts

def process_all_pdfs():
    print(f"--- Working Directory: {BASE_DIR} ---")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pdfs = list(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ Error: No PDFs found in {PDF_DIR}")
        return

    print(f"\n--- STEP 1: CONVERTING {len(pdfs)} PDFs TO IMAGES ---")
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        doc_out = OUT_DIR / doc_id
        doc_out.mkdir(parents=True, exist_ok=True)
        try:
            print(f"⏳ Converting {pdf_path.name}...")
            # باش نربحو الوقت: مابغيتيش يعاود يحول الصور يلا كانو ديجا كاينين؟
            if not list(doc_out.glob("*.png")):
                pages = convert_from_path(pdf_path, dpi=DPI, fmt="png", poppler_path=POPPLER_PATH)
                for i, page in enumerate(pages, start=1):
                    page_path = doc_out / f"page_{i:03d}.png"
                    if not page_path.exists():
                        page.save(page_path, "PNG")
                print(f"✅ {doc_id}: {len(pages)} pages processed.")
            else:
                print(f"✅ {doc_id}: Images already exist. Skipping conversion.")
        except Exception as e:
            print(f"❌ Error converting {pdf_path.name}: {e}")

    print("\n--- STEP 2: EXTRACTING TEXT (OCR) ---")
    for doc_folder in OUT_DIR.iterdir():
        if not doc_folder.is_dir(): continue
        
        pages_images = sorted(doc_folder.glob("*.png"))
        print(f"\n📂 Processing Document: {doc_folder.name}")

        for img_path in pages_images:
            json_name = f"{doc_folder.name}_{img_path.stem}_ocr.json"
            
            # باش مايعاودش يدير OCR للملفات لي ديجا خرجات
            if (OCR_OUT_DIR / json_name).exists(): 
                continue

            print(f"  📄 Reading {img_path.name}...")
            img = Image.open(img_path).convert("RGB")
            patches, coords = extract_patches(img)
            
            ocr_texts = ocr_patches(patches)

            ocr_data = {
                "source_pdf": doc_folder.name,
                "page_image": img_path.name,
                "data": []
            }
            
            for i, (bbox, text) in enumerate(zip(coords, ocr_texts)):
                if text and len(text) > 2: 
                    ocr_data["data"].append({
                        "patch_index": i,
                        "bbox": bbox,
                        "text": text
                    })

            with open(OCR_OUT_DIR / json_name, "w", encoding="utf-8") as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=4)
            print(f"  💾 Saved JSON for {img_path.name}")

    print("\n🎉 All tasks completed successfully!")

process_all_pdfs()

from langchain_community.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    """
    Download and return a Multilingual embedding model optimized for Arabic.
    """
    model_name = "intfloat/multilingual-e5-base"
    
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} 
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

embedding = download_embeddings()
print("✅ Embeddings model is ready for Arabic!")