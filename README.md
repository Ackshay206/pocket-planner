# 🏠 Dwell AI

**AI-Powered Interior Design Assistant** — A full-stack application that turns a simple floor plan image into a fully interactive, 3D-visualized, and shoppable interior design project.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![Gemini](https://img.shields.io/badge/Google-Gemini%20API-4285F4?logo=google)
![LangChain](https://img.shields.io/badge/LangGraph-Agentic-green)
![License](https://img.shields.io/badge/License-MIT-gray)

---

## 💡 What is Dwell AI?

Dwell AI is an intelligent interior design agent that solves the "blank canvas" problem. Users upload a photo of a floor plan or a room, and Dwell AI:
1.  **Understands** the space geometries using Computer Vision.
2.  **Generates** multiple creative layout options based on design principles.
3.  **Visualizes** the result in photorealistic 3D.
4.  **Refines** the design through a conversational chat interface.
5.  **Finds** real furniture products that match the generated design.

---

## 💎 How We Use Gemini API

![Dwell AI Architecture Diagram](./docs/dwellai-architecture.png)

At the heart of Dwell AI is a multi-agent system powered by distinct **Google Gemini** models, each specialized for a specific cognitive task. This isn't just a wrapper; it's a complex orchestration of vision, reasoning, and generation.

### 1. Vision Analysis Agent
*   **Model**: `gemini-3-pro-preview`
*   **Task**: "Digital Twin" Creation.
*   **Process**: The agent analyzes the uploaded image to spatially understand the room. It identifies:
    *   **Walls & Boundaries** (polygons)
    *   **Structural Elements** (Windows, Doors with aperture sizes)
    *   **Existing Furniture** (bounding boxes and types)
*   **Output**: Converts raw pixels into a structured JSON spatial graph.

### 2. Generative Designer Agent
*   **Model**: `gemini-2.5-pro`
*   **Task**: Spatial Reasoning & Planning.
*   **Process**: Acting as a professional interior designer, it takes the spatial graph and generates **3 distinct layout variations**:
    *   **Work Focused**: Maximizes productivity and natural light for desks.
    *   **Cozy & Relaxing**: Prioritizes social flow and comfort.
    *   **Creative & Bold**: Experiments with diagonal placements and flow.
*   **Highlight**: The model adheres to constraints (e.g., "don't block the door") while being creative.

### 3. Perspective Visualization Agent
*   **Model**: `gemini-2.5-flash-image`
*   **Task**: Photorealistic Rendering.
*   **Process**: It takes the 2D layout candidates and generates a photorealistic 3D perspective view from a specific camera angle, respecting the exact furniture positions and room style. This allows users to "feel" the space.

### 4. Conversational Editor Agent
*   **Model**: `gemini-2.5-pro`
*   **Task**: Natural Language Object Manipulation.
*   **Process**: Users can chat with the design (e.g., *"Move the desk next to the window"*). The agent understands the spatial intent, calculates the new coordinates, updates the JSON state, and triggers a re-render.

### 5. Smart Shopping Agent
*   **Model**: `gemini-2.5-pro` (Multimodal)
*   **Task**: Visual Search Query Generation.
*   **Process**: The agent looks at the *generated* 3D perspective image. It visually identifies furniture styles and decor (even items added creatively by the AI like plants or rugs) and generates precise, category-specific search queries for the **SerpAPI** (Google Shopping) to find real-world products.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Orchestration**: LangGraph (Stateful multi-agent workflow)
- **AI Models**: Google Gemini 3 Pro, Gemini 2.5 Pro, Gemini 2.5 Flash Image
- **Search**: SerpAPI (Google Shopping)
- **Geometry**: Shapely (Collision detection & spatial math)
- **Observability**: LangSmith

### Frontend
- **Framework**: Next.js 16 (React 19, App Router)
- **Styling**: Tailwind CSS 4
- **Visualization**: Konva (Canvas-based 2D floor plan rendering)
- **State**: React Hooks (Custom hooks for agent interaction)

---

## 🚀 Quick Start Guide

### Prerequisites
*   Python 3.11+
*   Node.js 20+
*   **Google AI API Key** (Get it [here](https://aistudio.google.com/apikey))
*   **SerpAPI Key** (Get it [here](https://serpapi.com/))

### 1. Clone & Install
```bash
git clone https://github.com/ackshay/dwell-ai.git
cd dwell-ai
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**Configure `.env`:**
```ini
GOOGLE_API_KEY=your_google_key
SERPAPI_KEY=your_serpapi_key
# Optional: For tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_key
```

**Start Server:**
```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env.local
```

**Configure `.env.local`:**
```ini
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Start Client:**
```bash
npm run dev
```
Visit `http://localhost:3000` to start designing!

---

## � API Endpoints

| Method | Endpoint | Description | Agent Used |
|--------|----------|-------------|------------|
| `POST` | `/api/v1/analyze` | Upload floor plan & detect objects | Vision Agent |
| `POST` | `/api/v1/optimize` | Generate 3 creative layout options | Designer Agent |
| `POST` | `/api/v1/render/perspective` | Generate 3D visualization | Perspective Agent |
| `POST` | `/api/v1/chat/edit` | Execute natural language edits | Editor Agent |
| `POST` | `/api/v1/shop` | Find products for the design | Shopping Agent |

---

## 📁 Project Structure

```text
dwell-ai/
├── backend/
│   ├── app/
│   │   ├── agents/                   # <--- INTELLIGENT AGENTS (LangGraph Nodes)
│   │   │   ├── vision_node.py        # Vision Agent: Analyzes floor plans using Gemini 3 Pro
│   │   │   ├── designer_node.py      # Designer Agent: Generates layouts using Gemini 2.5 Pro
│   │   │   ├── perspective_node.py   # Perspective Agent: Renders 3D views using Gemini 2.5 Flash Image
│   │   │   ├── shopping_node.py      # Shopping Agent: Finds products using Gemini Vision + SerpAPI
│   │   │   ├── chat_editor_node.py   # Editor Agent: Handles natural language design edits
│   │   │   └── graph.py              # LangGraph workflow definition & orchestration
│   │   ├── core/                     # <--- CORE LOGIC
│   │   │   ├── constraints.py        # Architectural rules & hard constraints
│   │   │   ├── geometry.py           # Shapely geometry operations & collision detection
│   │   │   └── scoring.py            # Layout scoring & evaluation engine
│   │   ├── models/                   # <--- DATA MODELS
│   │   │   ├── api.py                # API Request/Response schemas
│   │   │   ├── room.py               # Domain models (RoomObject, Constraints)
│   │   │   └── state.py              # LangGraph AgentState definitions
│   │   ├── routes/                   # <--- API ENDPOINTS
│   │   │   ├── analyze.py            # POST /analyze (Vision)
│   │   │   ├── optimize.py           # POST /optimize (Design)
│   │   │   ├── render.py             # POST /render (Perspective)
│   │   │   ├── chat.py               # POST /chat (Edit)
│   │   │   └── shop.py               # POST /shop (Shopping)
│   │   ├── tools/                    # <--- EXTERNAL TOOLS
│   │   │   ├── generate_image.py     # Gemini Imagen wrapper
│   │   │   ├── edit_image.py         # Gemini Image Editing wrapper
│   │   │   └── serp_search.py        # Google Shopping Search wrapper
│   │   ├── config.py                 # Configuration & Environment
│   │   └── main.py                   # FastAPI Application Entry
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/                      # <--- NEXT.JS APP ROUTER
│   │   │   ├── page.tsx              # Main Application Page
│   │   │   ├── layout.tsx            # Root Layout
│   │   │   └── globals.css           # Global Tailwind Styles
│   │   ├── components/               # <--- REACT COMPONENTS
│   │   │   ├── CanvasOverlay.tsx     # Interactive Konva Floor Plan
│   │   │   ├── ChatEditor.tsx        # Conversational Design Interface
│   │   │   ├── ImageUpload.tsx       # Drag & Drop Upload
│   │   │   ├── LayoutSelector.tsx    # Layout Variation Cards
│   │   │   ├── PerspectiveView.tsx   # 3D Visualization Viewer
│   │   │   ├── ProductRecommendations.tsx # Shopping List & Budget UI
│   │   │   ├── Sidebar.tsx           # Object List & Properties
│   │   │   ├── ObjectsPanel.tsx      # Draggable Object Palette
│   │   │   └── OptimizePanel.tsx     # Generation Controls
│   │   ├── hooks/                    # <--- CUSTOM HOOKS
│   │   │   ├── useAnalyze.ts         # Vision API Logic
│   │   │   ├── useOptimize.ts        # Layout Generation Logic
│   │   │   ├── usePerspective.ts     # Rendering Logic
│   │   │   ├── useChatEdit.ts        # Chat Logic
│   │   │   └── useShop.ts            # Shopping Logic
│   │   └── lib/                      # <--- UTILITIES
│   │       ├── api.ts                # Axios API Client
│   │       └── types.ts              # TypeScript Definitions
│   └── package.json
└── README.md
```

---

<p align="center">
  Made with ❤️ with GEMINI 3 API for better living spaces
</p>