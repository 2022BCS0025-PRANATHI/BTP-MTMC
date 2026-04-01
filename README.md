# BTP-MTMC: Multi-Target Multi-Camera Vehicle Tracking and Search

A modular, high-performance system for tracking vehicles across multiple asynchronous camera feeds. This project utilizes state-of-the-art computer vision models to enable natural language searching and cross-camera re-identification.

---

## 🚀 How it Works

The system operates in two main phases: **Indexing** and **Searching**.

### 1. Indexing Phase
During indexing, the system processes raw video files to build a searchable database (`tracks.json`):
- **Detection & Tracking**: Uses **YOLOv8** with the ByteTrack algorithm to detect vehicles (cars, trucks, buses, motorcycles) and maintain persistent track IDs within a single camera.
- **Feature Extraction**: 
    - **Visual Embeddings**: Uses **OpenAI's CLIP (ViT-B/32)** to extract high-dimensional visual features for each vehicle.
    - **Color Analysis**: Performs HSV-based color classification (e.g., "red", "silver").
    - **License Plate Recognition**: Employs a secondary YOLO model for plate detection and **PaddleOCR** for character recognition.
- **Context Extraction**: Extracts metadata such as location names and available on-screen text from the video OSD (On-Screen Display) using OCR.

### 2. Search & Re-ID Phase
Once indexed, you can query the database using natural language or images:
- **Query Parsing**: Translates text queries (e.g., *"red car near Petta"*) into structured constraints.
- **Similarity Matching**: Compares the query features against the index using a weighted similarity score (CLIP + Color + Aspect Ratio + Plate Similarity).
- **Cross-Camera Grouping**: Clusters matching tracks from different cameras to reconstruct the vehicle's path across the city.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- **NVIDIA GPU** (Highly Recommended for inference speed)
- CUDA & cuDNN installed

### 1. Clone the Repository
```bash
git clone https://github.com/2022BCS0025-PRANATHI/BTP-MTMC.git
cd BTP-MTMC
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Build the Track Index
Process your video files (e.g., `clip1.mp4`, `clip2.mp4`) to create the searchable database.
```bash
python main.py --videos clip1.mp4 clip2.mp4 --rebuild_index
```

### 2. Search the Database
You can search by typing a natural language query.
```bash
# Search by color and type
python main.py --query_text "red car"

# Search with location constraints
python main.py --query_text "white car near subashchandrabose"

# Search by license plate
python main.py --query_text "KL07CQ3939"
```

### 3. View Results
After searching, the system generates a visual report in the `results_out/` directory. Open **`results_out/results.html`** in any web browser to see the ranked matches and cross-camera groups.

---

## 📂 Project Structure

- `mtmc/`: Core package containing modular logic for OCR, Embeddings, and Search.
- `main.py`: CLI entry point.
- `requirements.txt`: List of Python dependencies.
- `track_index/`: (Auto-generated) Stores the processed vehicle database.
- `results_out/`: (Auto-generated) Stores search reports and cropped images.

---

## 📜 License
This project is developed as part of the BTP (Bachelor Thesis Project).
