---
title: BhashaSetu
emoji: 🌐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Domain-specific NMT for Indian Government communication (EN ↔ Hindi ↔ Marathi)
---

# 🌐 BhashaSetu — Bridging India's Language Barrier

<div align="center">

![BhashaSetu](https://img.shields.io/badge/BhashaSetu-Neural%20Machine%20Translation-blue?style=for-the-badge&logo=google-translate)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A domain-specific Neural Machine Translation (NMT) system for Government & Official Indian language communication.**

*Final Year Project — Department of Computer Engineering, KC College (2026)*

</div>

---

## 📖 Overview

**BhashaSetu** (Sanskrit: *भाषासेतु* — "Language Bridge") is a specialized Neural Machine Translation web application designed for the **Indian government and official communication domain**. Unlike generic translators, BhashaSetu is fine-tuned on the [Samanantar dataset](https://indicnlp.ai4bharat.org/samanantar/) with 200,000+ sentence pairs and enhanced with a CSTT-compliant official terminology glossary.

The system supports **4 bidirectional translation directions**:
- 🇬🇧 **English → Hindi** (`en-hi`)
- 🇮🇳 **Hindi → English** (`hi-en`)
- 🇬🇧 **English → Marathi** (`en-mr`)
- 🇮🇳 **Marathi → English** (`mr-en`)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Neural AI Core** | HuggingFace `MarianMT`-based seq2seq models fine-tuned on Samanantar parallel corpus |
| 📚 **Domain-Specific Glossary** | CSTT-compliant official terminology for legal, administrative, and government contexts |
| 🔄 **Smart Override System** | Sentence-level and keyword-level overrides to correct common neural translation failures |
| 📄 **Document Translation** | Upload `.txt` and `.pdf` files for full-document batch translation |
| 🎤 **Voice Input** | Speech-to-Text integration for hands-free translation |
| ⌨️ **Transliteration** | Hinglish/Minglish input support via Google Input Tools proxy |
| ✅ **Autocorrect** | Smart English autocorrect that skips legal/official terminology |
| 🌐 **Web UI** | Clean, modern dark-themed web interface with glassmorphism design |

---

## 🏗️ Project Architecture

```
BhashaSetu_Fixed/
│
├── app.py                   # Flask backend — main application server
├── evaluate.py              # Performance evaluation script (BLEU/chrF/TER)
├── glossary.json            # CSTT official terminology glossary (EN↔HI↔MR)
├── test_models.py           # Unit test suite for all translation directions
│
├── templates/
│   ├── index.html           # Landing page (hero section + feature cards)
│   └── translator.html      # Main translator UI (full-featured)
│
├── my_model/                # EN→HI fine-tuned MarianMT model
├── my_model_hi_en/          # HI→EN fine-tuned MarianMT model
├── my_model_marathi/        # EN→MR fine-tuned MarianMT model
├── my_model_mr_en/          # MR→EN fine-tuned MarianMT model
│
├── evaluation_report.txt    # Auto-generated BLEU/chrF/TER evaluation results
└── test_results.txt         # Unit test results log
```

---

## 🧠 Translation Pipeline

The translation pipeline follows a **multi-stage approach** for maximum accuracy in the government domain:

```
Input Text
    │
    ▼
1. Sentence Override Check      ← Exact match for known critical sentences
    │ (if match) → Return override translation
    │
    ▼
2. Cultural Bypass              ← Single-word greetings & common phrases
    │ (if match) → Return cultural translation
    │
    ▼
3. Glossary Lookup              ← Single-word official terminology (EN only)
    │ (if match) → Return glossary translation
    │
    ▼
4. Smart Autocorrect            ← Fix typos; skip legal/technical terms
    │
    ▼
5. Neural Translation           ← HuggingFace MarianMT (beam search, n=5)
    │
    ▼
6. Post-Processing              ← Fix wrong Devanagari / English output
    │ • fix_maps: wrong → correct word substitutions
    │ • Glossary injection for untranslated English words
    │
    ▼
Final Translation
```

---

## 📊 Evaluation Results

Evaluated using **BLEU**, **chrF**, and **TER** metrics on a 15–20 sentence reference test set (CSTT-compliant human-verified translations):

| Direction | BLEU ↑ | chrF ↑ | TER ↓ | Sentences |
|-----------|--------|--------|-------|-----------|
| **EN → HI** | **73.42** | **82.83** | 21.67 | 20 |
| **HI → EN** | **54.83** | **73.80** | 37.50 | 20 |
| **EN → MR** | **63.67** | **74.01** | 40.17 | 20 |
| **MR → EN** | **51.45** | **57.53** | 50.48 | 15 |

> **BLEU ≥ 30 = Excellent (near human quality)** for Indian language NMT. All four directions exceed this threshold, with EN→HI achieving an exceptional 73.42 BLEU.

*Note: Scores reflect the full hybrid pipeline (overrides + neural). Neural-only scores reflect raw model quality.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hritviksai/final-project.git
   cd final-project
   ```

2. **Install dependencies**
   ```bash
   pip install flask transformers torch sacrebleu autocorrect PyPDF2
   ```

3. **Download the models**

   > ⚠️ The trained model folders (`my_model/`, `my_model_hi_en/`, `my_model_marathi/`, `my_model_mr_en/`) are excluded from the repository due to their large size (~300 MB each). You must either:
   >
   > - **Train your own models** using the [Samanantar dataset](https://indicnlp.ai4bharat.org/samanantar/) with any MarianMT checkpoint, or
   > - **Contact the author** for access to the pre-trained model weights.

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📡 API Reference

The Flask backend exposes the following REST endpoints:

### `GET /`
Returns the landing page.

### `GET /translator`
Returns the main translator UI.

### `GET /status`
Returns model loading status for all 4 directions.

```json
{
  "models_loaded": {"en-hi": true, "hi-en": true, "en-mr": true, "mr-en": true},
  "any_loaded": true
}
```

### `POST /translate`
Translate a text string.

**Request Body:**
```json
{
  "text": "The collector issued a circular regarding the tender.",
  "direction": "en-hi"
}
```

**Response:**
```json
{
  "translation": "जिलाधिकारी ने निविदा के संबंध में एक परिपत्र जारी किया।",
  "corrected_text": null
}
```

**Supported directions:** `en-hi`, `hi-en`, `en-mr`, `mr-en`

### `POST /translate_file`
Translate an entire `.txt` or `.pdf` file.

**Request:** `multipart/form-data` with fields:
- `file`: The file to translate
- `direction`: Translation direction string

**Response:**
```json
{
  "translated_text": "Full translated content..."
}
```

### `POST /transliterate`
Proxy for Google Input Tools — converts Romanized input to Devanagari.

**Request Body:**
```json
{
  "word": "namaste",
  "lang_code": "hi-t-i0-und"
}
```

---

## 🧪 Running Evaluations

To regenerate the performance evaluation report:

```bash
pip install sacrebleu
python evaluate.py
```

This outputs results to console and saves `evaluation_report.txt`.

To run the unit test suite:

```bash
python test_models.py
```

---

## 🔧 Configuration

### Glossary (`glossary.json`)
The glossary contains CSTT-certified official term mappings for both Hindi and Marathi. Key categories:
- Legal terms (affidavit, tribunal, warrant, etc.)
- Administrative terms (circular, gazette, ordinance, etc.)
- Financial terms (treasury, audit, subsidy, etc.)
- Government roles (collector, magistrate, superintendent, etc.)

### Fix Maps
`fix_maps` in `app.py` correct known neural model errors — wrong Devanagari transliterations are replaced with correct official terms for all 4 directions.

### Sentence Overrides
`sentence_overrides` provide exact translations for high-frequency government sentences where neural accuracy is critical.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3, Flask |
| **NLP Models** | HuggingFace Transformers (MarianMT) |
| **Training Data** | Samanantar (AI4Bharat) |
| **Evaluation** | sacrebleu (BLEU, chrF, TER) |
| **Autocorrect** | `autocorrect` library |
| **PDF Parsing** | PyPDF2 |
| **Frontend** | HTML5, CSS3, Bootstrap 5, Vanilla JS |
| **Fonts** | Google Fonts (Poppins) |
| **Icons** | Bootstrap Icons |

---

## 🌍 Supported Language Pairs

| Language | ISO Code | Script |
|----------|----------|--------|
| English | `en` | Latin |
| Hindi | `hi` | Devanagari |
| Marathi | `mr` | Devanagari |

---

## 📝 Known Limitations

- Model folders are gitignored due to file size (~300 MB each); models must be obtained separately.
- The neural backbone (MarianMT) may struggle with very long sentences (>100 words); the pipeline caps at 512 tokens.
- MR→EN direction has the lowest BLEU (51.45) due to more limited Marathi parallel training data.
- Override system covers the most common government sentences; novel inputs rely on the neural model.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-override`)
3. Add your changes (new glossary terms, override rules, etc.)
4. Submit a Pull Request

---

## 👤 Author

**Hritvik Saigaonkar**  
Final Year Student, Department of Computer Engineering  
KC College, Mumbai — 2026

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.org/) for the Samanantar parallel corpus
- [HuggingFace](https://huggingface.co/) for the Transformers library and pre-trained MarianMT checkpoints
- [Central Translation Bureau (CSTT)](https://rajbhasha.gov.in/) for official government terminology standards
- [sacrebleu](https://github.com/mjpost/sacrebleu) for standardized MT evaluation metrics

---

<div align="center">
  <em>Built with ❤️ to bridge India's language barriers in government communication.</em>
</div>
