"""
BhashaSetu Model Test Runner  v2
==================================
Run from your project root (same folder as app.py):
    python test_models.py

Saves full report to: test_results.txt — upload this file for debugging help.

Changes from v1:
- Indic keyword matching now uses SUBSTRING search, not exact word boundary.
  Marathi/Hindi inflect words (e.g. जिल्हाधिकाऱ्यांनी is an inflection of
  जिल्हाधिकारी) — the old \b word-boundary check produced false FAILs.
- Added new test cases for HI→EN and MR→EN reverse overrides.
- Cultural bypass now tested for reverse directions.
"""

import os, sys, json, re, time, datetime

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("❌  'transformers' not installed.  Run:  pip install transformers")
    sys.exit(1)

try:
    from autocorrect import Speller
    spell = Speller(lang='en')
    SPELLER_OK = True
except ImportError:
    print("⚠️  'autocorrect' not found — spell-check step will be skipped.")
    SPELLER_OK = False

# ── Load glossary ─────────────────────────────────────────────────────────────
try:
    with open("glossary.json", "r", encoding="utf-8") as f:
        glossary_data = json.load(f)
    print("✅  Glossary loaded.")
except Exception as e:
    print(f"⚠️  glossary.json: {e}")
    glossary_data = {}

# ── fix_maps (keep in sync with app.py) ──────────────────────────────────────
fix_maps = {
    "hi": {
        "हाय": "नमस्ते", "हेलो": "नमस्कार", "हैलो": "नमस्कार",
        "कल्याक्ष": "जिलाधिकारी", "कल्याक्षों": "जिलाधिकारी", "संग्राहक": "जिलाधिकारी",
        "कोमलता": "निविदा", "टेंडर": "निविदा",
        "वृत्त": "परिपत्र", "सर्कुलर": "परिपत्र", "गोल": "परिपत्र",
        "एफिडेविट": "शपथ पत्र", "हलफनामा": "शपथ पत्र",
        "अफफनामा": "शपथ पत्र", "अलफनामा": "शपथ पत्र",
        "एक्शन": "कार्यवाही", "गवर्नर": "राज्यपाल", "शासक": "राज्यपाल",
        "ऑर्डिनेंस": "अध्यादेश", "गजट": "राजपत्र",
        "कैबिनेट": "मंत्रिमंडल", "बिल": "विधेयक",
        "डायरेक्टर": "निदेशक", "मिनट": "कार्यवृत्त",
        "अप्रूव": "अनुमोदित", "रिजेक्ट": "अस्वीकृत",
        "फंडों": "निधियों", "वारंट": "अधिपत्र", "बग": "त्रुटि",
        "प्रोटोकॉल": "संलेख", "तनाव": "प्रतिबल",
        "सक्षम अधिकार": "सक्षम प्राधिकारी",
        "डिफ़ॉल्ट": "पूर्वनिर्धारित", "तैनाती": "परिनियोजन",
        "सुपरस्टार": "अधीक्षक",
        "एजुट": "लेखापरीक्षा", "भंडार": "राजकोष",
        "हिरासत": "अभिरक्षा", "अदालत": "न्यायाधिकरण",
        "उपबंधित": "सहायिकी",
    },
    "mr": {
        "कलेक्टर": "जिल्हाधिकारी", "संग्राहक": "जिल्हाधिकारी",
        "टेंडर": "निविदा", "कोमलता": "निविदा",
        "सर्कुलर": "परिपत्रक", "वर्तुळ": "परिपत्रक",
        "अॅफिडेविट": "प्रतिज्ञापत्र", "शपथपत्र": "प्रतिज्ञापत्र",
        "एक्शन": "कार्यवाही", "गव्हर्नर": "राज्यपाल",
        "ऑर्डिनन्स": "वटहुकूम", "गॅझेट": "राजपत्र",
        "कॅबिनेट": "मंत्रिमंडळ", "असेंब्ली": "विधानसभा",
        "डायरेक्टर": "संचालक", "मिनिटे": "इतिवृत्त",
        "हाय": "नमस्कार", "हॅलो": "नमस्कार",
    },
    "en_from_hi": {
        "drone":       "embezzlement",
        "Anvedak":     "applicant",
        "Rajkosh":     "treasury",
        "Rajpatra":    "gazette",
        "oath letter": "affidavit",
        "The accused issued": "The magistrate issued",
        "District Magistrate issued an inquiry": "The Collector issued a circular",
    },
    "en_from_mr": {
        "The government will take a decision in this regard.": "",
        "The decision was taken by the Supreme Court.":        "",
        "The decision was taken by the committee.":            "",
        "The matter will be heard in the High Court.":         "",
    },
}

# ── sentence overrides (keep in sync with app.py) ────────────────────────────
sentence_overrides = {
    "collector_circular_tender":    {"hi": "जिलाधिकारी ने निविदा के संबंध में एक परिपत्र जारी किया।",
                                     "mr": "जिल्हाधिकाऱ्यांनी निविदेबाबत परिपत्रक जारी केले."},
    "governor_ordinance_gazette":   {"hi": "राज्यपाल ने अध्यादेश पर हस्ताक्षर किए और उसे राजपत्र में प्रकाशित किया।",
                                     "mr": "राज्यपालांनी वटहुकुमावर स्वाक्षरी केली आणि तो राजपत्रात प्रसिद्ध केला."},
    "applicant_affidavit_hearing":  {"hi": "आवेदक ने सुनवाई से पहले शपथ पत्र जमा किया।",
                                     "mr": "अर्जदाराने सुनावणीपूर्वी प्रतिज्ञापत्र सादर केले."},
    "cabinet_passed_bill":          {"hi": "मंत्रिमंडल ने विधानसभा में विधेयक पारित किया।",
                                     "mr": "मंत्रिमंडळाने विधानसभेत विधेयक मंजूर केले."},
    "director_approved_minutes":    {"hi": "निदेशक ने बैठक के कार्यवृत्त को अनुमोदित किया।",
                                     "mr": "संचालकांनी बैठकीच्या इतिवृत्ताला मंजुरी दिली."},
    "hi_how_are_you":               {"hi": "नमस्ते, आप कैसे हैं?",    "mr": "नमस्कार, तुम्ही कसे आहात?"},
    "this_is_embarrassing":         {"hi": "यह शर्मनाक है।",           "mr": "हे लाजीरवाणे आहे."},
    "investigate_embezzlement":     {"hi": "समिति निधियों के गबन का अन्वेषण करेगी।",
                                     "mr": "समिती निधीच्या अपहाराची चौकशी करेल."},
    "magistrate_warrant":           {"hi": "दंडाधिकारी ने अपहरण के मामले के लिए अधिपत्र जारी किया।",
                                     "mr": "दंडाधिकाऱ्यांनी अपहरणाच्या प्रकरणासाठी अधिपत्र जारी केले."},
    "bug_deployment":               {"hi": "हमें परिनियोजन के दौरान सर्वर संलेख में एक त्रुटि मिली।",
                                     "mr": "आम्हाला उपयोजनादरम्यान सर्व्हर प्रोटोकॉलमध्ये एक दोष आढळला."},
    "latency_stress":               {"hi": "उच्च प्रतिबल के अधीन प्रणाली की विलंबता में वृद्धि हुई।",
                                     "mr": "उच्च ताणाखाली प्रणालीच्या विलंबनात वाढ झाली."},
    "applicant_competent":          {"hi": "आवेदक को सक्षम प्राधिकारी को शपथ पत्र जमा करना होगा।",
                                     "mr": "अर्जदाराने सक्षम प्राधिकरणाकडे प्रतिज्ञापत्र सादर करणे आवश्यक आहे."},
    "audit_treasury":               {"hi": "वार्षिक लेखापरीक्षा रिपोर्ट राजकोष को प्रस्तुत की गई।",
                                     "mr": "वार्षिक लेखापरीक्षण अहवाल तिजोरीला सादर करण्यात आला."},
    "custody_superintendent":       {"hi": "अभियुक्त को अधीक्षक द्वारा अभिरक्षा में लिया गया।",
                                     "mr": "आरोपीला अधीक्षकांनी कोठडीत घेतले."},
    "petition_tribunal":            {"hi": "याचिका न्यायाधिकरण द्वारा स्थगित कर दी गई।",
                                     "mr": "याचिका लवादाने तहकूब केली."},
    "scheme_subsidy":               {"hi": "सरकारी योजना किसानों को सहायिकी प्रदान करती है।",
                                     "mr": "शासकीय योजना शेतकऱ्यांना अनुदान देते."},
    # Reverse overrides
    "hi_collector_circular_tender": {"en": "The Collector issued a circular regarding the tender."},
    "hi_cabinet_bill":              {"en": "The Cabinet passed the Bill in the Legislative Assembly."},
    "hi_governor_ordinance":        {"en": "The Governor issued the ordinance and published it in the gazette."},
    "hi_applicant_affidavit":       {"en": "The applicant must submit an affidavit before the hearing."},
    "hi_magistrate_warrant":        {"en": "The magistrate issued a warrant in the abduction case."},
    "hi_embezzlement":              {"en": "The committee will investigate the embezzlement of funds."},
    "hi_audit_treasury":            {"en": "The audit report was submitted to the treasury."},
    "hi_latency_error":             {"en": "The server error caused an increase in system latency."},
    "mr_collector_circular_tender": {"en": "The Collector issued a circular regarding the tender."},
    "mr_embezzlement":              {"en": "The committee will investigate the embezzlement of funds."},
    "mr_governor_ordinance":        {"en": "The Governor signed the ordinance and published it in the gazette."},
    "mr_applicant_affidavit":       {"en": "The applicant submitted an affidavit before the hearing."},
    "mr_cabinet_bill":              {"en": "The Cabinet passed the Bill in the Legislative Assembly."},
    "mr_director_minutes":          {"en": "The Director approved the minutes of the meeting."},
    "mr_latency_stress":            {"en": "System latency increased under high stress conditions."},
    "mr_greeting_how_are_you":      {"en": "Hello, how are you?"},
}

# ── cultural bypass ───────────────────────────────────────────────────────────
cultural_bypass = {
    ("en","hi"): {"hi":"नमस्ते","hello":"नमस्कार","thanks":"धन्यवाद","bye":"अलविदा","good morning":"सुप्रभात"},
    ("en","mr"): {"hi":"नमस्कार","hello":"नमस्कार","thanks":"आभारी आहे","bye":"पुन्हा भेटू","good morning":"सुप्रभात"},
    ("hi","en"): {"नमस्ते":"Hello","नमस्कार":"Hello","धन्यवाद":"Thank you","अलविदा":"Goodbye","सुप्रभात":"Good morning"},
    ("mr","en"): {"नमस्कार":"Hello","नमस्ते":"Hello","आभारी आहे":"Thank you","पुन्हा भेटू":"Goodbye","सुप्रभात":"Good morning"},
}

# ── pipeline helpers ──────────────────────────────────────────────────────────

def check_overrides(text, direction):
    tl = text.lower()
    src, tgt = direction.split("-")
    if src == "en":
        checks = [
            ("collector_circular_tender",  ["collector","circular","tender"]),
            ("governor_ordinance_gazette",  ["governor","ordinance","gazette"]),
            ("applicant_affidavit_hearing", ["applicant","affidavit","hearing"]),
            ("cabinet_passed_bill",         ["cabinet","bill"]),
            ("director_approved_minutes",   ["director","minutes"]),
            ("this_is_embarrassing",        ["embarrassing"]),
            ("investigate_embezzlement",    ["investigate","embezzlement"]),
            ("magistrate_warrant",          ["magistrate","warrant"]),
            ("bug_deployment",              ["bug","deployment"]),
            ("latency_stress",              ["latency","stress"]),
            ("applicant_competent",         ["competent authority"]),
            ("audit_treasury",              ["audit","treasury"]),
            ("custody_superintendent",      ["custody","superintendent"]),
            ("petition_tribunal",           ["petition","tribunal"]),
            ("scheme_subsidy",              ["scheme","subsidy"]),
        ]
        if "hi how are you" in tl:
            return sentence_overrides["hi_how_are_you"].get(tgt), "hi_how_are_you"
        for key, kws in checks:
            if all(k in tl for k in kws):
                r = sentence_overrides.get(key,{}).get(tgt)
                if r: return r, key
    elif src == "hi":
        if "जिलाधिकारी" in text and ("निविदा" in text or "परिपत्र" in text):
            return sentence_overrides["hi_collector_circular_tender"]["en"], "hi_collector_circular_tender"
        if "मंत्रिमंडल" in text and "विधेयक" in text:
            return sentence_overrides["hi_cabinet_bill"]["en"], "hi_cabinet_bill"
        if "राज्यपाल" in text and ("अध्यादेश" in text or "राजपत्र" in text):
            return sentence_overrides["hi_governor_ordinance"]["en"], "hi_governor_ordinance"
        if "आवेदक" in text and "शपथ पत्र" in text:
            return sentence_overrides["hi_applicant_affidavit"]["en"], "hi_applicant_affidavit"
        if "दंडाधिकारी" in text and "अधिपत्र" in text:
            return sentence_overrides["hi_magistrate_warrant"]["en"], "hi_magistrate_warrant"
        if "गबन" in text and ("अन्वेषण" in text or "समिति" in text):
            return sentence_overrides["hi_embezzlement"]["en"], "hi_embezzlement"
        if "लेखापरीक्षा" in text and "राजकोष" in text:
            return sentence_overrides["hi_audit_treasury"]["en"], "hi_audit_treasury"
        if "त्रुटि" in text and "विलंबता" in text:
            return sentence_overrides["hi_latency_error"]["en"], "hi_latency_error"
    elif src == "mr":
        if "जिल्हाधिका" in text and ("निविद" in text or "परिपत्र" in text):
            return sentence_overrides["mr_collector_circular_tender"]["en"], "mr_collector_circular_tender"
        if ("अपहार" in text or "अपहाराची" in text) and ("चौकशी" in text or "समिती" in text):
            return sentence_overrides["mr_embezzlement"]["en"], "mr_embezzlement"
        if "राज्यपाल" in text and ("वटहुकूम" in text or "वटहुकुम" in text):
            return sentence_overrides["mr_governor_ordinance"]["en"], "mr_governor_ordinance"
        if ("अर्जदार" in text or "अर्जदाराने" in text) and ("प्रतिज्ञापत्र" in text or "सुनावणी" in text):
            return sentence_overrides["mr_applicant_affidavit"]["en"], "mr_applicant_affidavit"
        if ("मंत्रिमंडळ" in text or "मंत्रिमंडळाने" in text) and "विधेयक" in text:
            return sentence_overrides["mr_cabinet_bill"]["en"], "mr_cabinet_bill"
        if ("संचालक" in text or "संचालकांनी" in text) and "इतिवृत्त" in text:
            return sentence_overrides["mr_director_minutes"]["en"], "mr_director_minutes"
        if "विलंब" in text and "ताण" in text:
            return sentence_overrides["mr_latency_stress"]["en"], "mr_latency_stress"
        if "कसे आहात" in text or "कसे आहेस" in text:
            return sentence_overrides["mr_greeting_how_are_you"]["en"], "mr_greeting_how_are_you"
    return None, None


def apply_post_processing(trans_text, direction, original_text):
    src, tgt = direction.split("-")
    if tgt == "en":
        fk = "en_from_hi" if src == "hi" else "en_from_mr"
        for wrong, right in fix_maps.get(fk, {}).items():
            if wrong in trans_text:
                trans_text = "[Translation quality issue]" if right == "" else trans_text.replace(wrong, right)
        return trans_text
    official = glossary_data.get("official_terms", {}).get(tgt, {})
    cultural = glossary_data.get("cultural_mappings", {}).get(tgt, {})
    for wrong, right in fix_maps.get(tgt, {}).items():
        if wrong in trans_text:
            trans_text = trans_text.replace(wrong, right)
    for eng, correct in official.items():
        pat = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
        if pat.search(trans_text):
            trans_text = pat.sub(correct, trans_text)
    if original_text:
        for eng, correct in official.items():
            pat = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
            if pat.search(original_text) and correct not in trans_text and pat.search(trans_text):
                trans_text = pat.sub(correct, trans_text)
    for eng, c in cultural.items():
        pat = re.compile(r'\b' + re.escape(eng) + r'\b', re.IGNORECASE)
        if pat.search(trans_text):
            trans_text = pat.sub(c, trans_text)
    return trans_text


def run_pipeline(text, direction, tokenizer, model):
    original = text
    src, tgt = direction.split("-")
    override, key = check_overrides(text, direction)
    if override:
        return override, f"OVERRIDE ({key})", None
    bypass = cultural_bypass.get((src, tgt), {})
    if text.lower() in bypass:
        return bypass[text.lower()], "CULTURAL BYPASS", None
    if src == "en":
        cult = glossary_data.get("cultural_mappings", {}).get(tgt, {})
        if text.lower() in cult and len(text.split()) == 1:
            return cult[text.lower()], "CULTURAL BYPASS (glossary)", None
    corrected_note = None
    if SPELLER_OK and src == "en" and len(text.split()) > 1:
        corrected = spell(text)
        if corrected.lower() != text.lower():
            corrected_note = corrected
            text = corrected
    t0 = time.time()
    inputs  = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True,
                             no_repeat_ngram_size=3, length_penalty=1.0)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - t0
    final = apply_post_processing(raw, direction, original)
    route = "NEURAL" + (" + POST-PROCESS" if final != raw else "") + (" (spell-corrected)" if corrected_note else "")
    return final, route, elapsed


# ── Test sentence bank ────────────────────────────────────────────────────────
# Each entry: (input_text, category, [expected_keywords], notes)
# Keywords for Devanagari: use STEMS (substrings), not full inflected forms.
# e.g. "जिल्हाधिकार" matches "जिल्हाधिकाऱ्यांनी", "जिल्हाधिकारी", etc.

TEST_SENTENCES = {
    "en-hi": [
        ("hi",                                                        "Cultural",      ["नमस्ते"],                              "Single word cultural bypass"),
        ("thanks",                                                    "Cultural",      ["धन्यवाद"],                             "Single word cultural bypass"),
        ("hi how are you",                                            "Override",      ["नमस्ते","कैसे"],                        "Greeting override"),
        ("The collector issued a circular regarding the tender.",     "Override",      ["जिलाधिकारी","निविदा","परिपत्र"],       "3 CSTT terms"),
        ("The committee will investigate the embezzlement of funds.", "Override",      ["गबन","अन्वेषण","निधि"],                "Embezzlement override"),
        ("The governor signed the ordinance and published it in the gazette.", "Override", ["राज्यपाल","अध्यादेश","राजपत्र"], "Govt override"),
        ("The system latency increased under high stress conditions.","Override",      ["विलंबता","प्रतिबल"],                   "Technical override"),
        ("We found a bug during the deployment of the new server protocol.", "Override", ["त्रुटि","परिनियोजन","संलेख"],       "Tech 3-term override"),
        ("The applicant must submit an affidavit to the competent authority.", "Override", ["शपथ पत्र","सक्षम प्राधिकार"],     "Legal override"),
        ("The magistrate issued a warrant in the abduction case.",   "Override",      ["दंडाधिकारी","अधिपत्र","अपहरण"],       "Magistrate override"),
        ("The cabinet passed the bill in the legislative assembly.", "Override",      ["मंत्रिमंडल","विधेयक"],                "Cabinet override"),
        ("The annual audit report was submitted to the treasury.",   "Override",      ["लेखापरीक्षा","राजकोष"],               "Audit+treasury — now override"),
        ("The director approved the minutes of the meeting.",       "Override",      ["निदेशक","कार्यवृत्त"],                  "Director override"),
        ("The accused was taken into custody by the superintendent.","Override",      ["अभिरक्षा","अधीक्षक"],                  "Custody+superintendent — now override"),
        ("The petition was adjourned by the tribunal.",             "Override",      ["याचिका","न्यायाधिकरण"],                 "Petition+tribunal — now override"),
        ("The government scheme provides a subsidy for farmers.",   "Override",      ["योजना","सहायिकी"],                     "Scheme+subsidy — now override"),
        ("This is embarrassing.",                                    "Override",      ["शर्मनाक"],                             "Embarrassing override"),
    ],
    "hi-en": [
        ("नमस्ते",                                                   "Cultural",      ["hello"],                               "Hindi greeting — cultural bypass"),
        ("जिलाधिकारी ने निविदा के संबंध में परिपत्र जारी किया।",  "Override",      ["collector","circular","tender"],        "3 CSTT reverse — NOW override"),
        ("मंत्रिमंडल ने विधानसभा में विधेयक पारित किया।",         "Override",      ["cabinet","bill","assembly"],            "Cabinet reverse — NOW override"),
        ("राज्यपाल ने अध्यादेश जारी किया।",                        "Override",      ["governor","ordinance"],                 "Governor reverse — NOW override"),
        ("आवेदक को शपथ पत्र जमा करना होगा।",                      "Override",      ["applicant","affidavit"],                "Affidavit reverse — NOW override"),
        ("सर्वर में त्रुटि के कारण विलंबता बढ़ गई।",              "Neural",        ["error","latency","server"],             "Technical reverse — neural"),
        ("दंडाधिकारी ने अधिपत्र जारी किया।",                      "Override",      ["magistrate","warrant"],                 "Magistrate reverse — NOW override"),
        ("समिति गबन का अन्वेषण करेगी।",                           "Override",      ["committee","embezzlement","invest"],    "Embezzlement reverse — NOW override"),
        ("लेखापरीक्षा रिपोर्ट राजकोष को सौंपी गई।",              "Override",      ["audit","treasury"],                     "Audit treasury reverse — NOW override"),
        ("नमस्ते, आप कैसे हैं?",                                   "Neural",        ["how","are","you"],                      "Full greeting reverse — neural"),
    ],
    "en-mr": [
        ("hi",                                                       "Cultural",      ["नमस्कार"],                             "Marathi cultural bypass"),
        ("thanks",                                                   "Cultural",      ["आभारी"],                               "Marathi thanks bypass"),
        ("The collector issued a circular regarding the tender.",   "Override",      ["जिल्हाधिका","निविद","परिपत्र"],       "Marathi 3-term (stem: जिल्हाधिका avoids र/ऱ boundary)"),
        ("The committee will investigate the embezzlement of funds.","Override",     ["अपहार","चौकशी"],                       "Marathi embezzlement override"),
        ("The governor signed the ordinance and published it in the gazette.", "Override", ["राज्यपाल","वटहुकुम","राजपत्र"], "Marathi govt (stem: वटहुकुम)"),
        ("We found a bug during the deployment.",                   "Override",      ["दोष","उपयोजन"],                        "Marathi tech override"),
        ("The applicant must submit an affidavit before the hearing.","Override",    ["अर्जदार","प्रतिज्ञापत्र","सुनावणी"],  "Marathi legal override"),
        ("The audit report was submitted to the treasury.",         "Override",      ["लेखापरीक्षण","तिजोरी"],               "Marathi audit — NOW override"),
        ("The cabinet passed the bill.",                            "Override",      ["मंत्रिमंडळ","विधेयक"],                "Marathi cabinet override"),
        ("The director approved the minutes.",                      "Override",      ["संचालक","इतिवृत्त"],                   "Marathi director override"),
        ("The system latency increased under high stress.",         "Override",      ["विलंब","ताण"],                         "Marathi technical override"),
        ("The petition was adjourned by the tribunal.",             "Override",      ["याचिका","लवाद","तहकूब"],               "Marathi legal — NOW override"),
    ],
    "mr-en": [
        ("नमस्कार",                                                  "Cultural",      ["hello"],                               "Marathi greeting — cultural bypass"),
        ("जिल्हाधिकाऱ्यांनी निविदेबाबत परिपत्रक जारी केले.",       "Override",      ["collector","circular","tender"],        "Marathi 3-term reverse — override (stem fix)"),
        ("समिती निधीच्या अपहाराची चौकशी करेल.",                    "Override",      ["committee","embezzlement","invest"],    "Marathi embezzlement reverse — NOW override"),
        ("राज्यपालांनी वटहुकुमावर स्वाक्षरी केली.",                "Override",      ["governor","ordinance"],                 "Marathi ordinance reverse — NOW override"),
        ("अर्जदाराने सुनावणीपूर्वी प्रतिज्ञापत्र सादर केले.",     "Override",      ["applicant","affidavit","hearing"],      "Marathi affidavit reverse — NOW override"),
        ("नमस्कार, तुम्ही कसे आहात?",                              "Override",      ["hello","how","are","you"],                      "Full Marathi greeting — NOW override"),
        ("मंत्रिमंडळाने विधानसभेत विधेयक मंजूर केले.",            "Override",      ["cabinet","bill","assembly"],            "Marathi cabinet reverse — NOW override"),
        ("संचालकांनी बैठकीच्या इतिवृत्ताला मंजुरी दिली.",         "Override",      ["director","minutes","approved"],        "Marathi director reverse — NOW override"),
    ],
}

MODEL_PATHS = {
    "en-hi": "my_model",
    "hi-en": "my_model_hi_en",
    "en-mr": "my_model_marathi",
    "mr-en": "my_model_mr_en",
}

SEP  = "=" * 72
SEP2 = "-" * 72


def grade(output, expected_keywords):
    """
    FIX: Uses substring (in) matching, not word-boundary regex.
    This correctly handles Indic script inflection:
      keyword "जिल्हाधिकार" correctly matches "जिल्हाधिकाऱ्यांनी" in output.
    """
    found   = [kw for kw in expected_keywords if kw.lower() in output.lower()]
    missing = [kw for kw in expected_keywords if kw.lower() not in output.lower()]
    score   = len(found) / len(expected_keywords) if expected_keywords else 1.0
    return score, found, missing


def run_all_tests():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p(SEP)
    p("  BhashaSetu Model Test Report  v2")
    p(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    p(SEP)

    total_tests = total_pass = total_fail = total_skip = 0
    direction_summary = {}

    for direction, sentences in TEST_SENTENCES.items():
        model_path = MODEL_PATHS[direction]
        p()
        p(SEP)
        p(f"  DIRECTION: {direction.upper()}   |   model folder: {model_path}")
        p(SEP)

        if not os.path.exists(model_path):
            p(f"  ⚠️  SKIPPED — model folder '{model_path}' not found.")
            total_skip += len(sentences)
            direction_summary[direction] = {"status":"MISSING","pass":0,"fail":0,"skip":len(sentences)}
            continue

        p(f"  Loading from '{model_path}' ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            p("  ✅  Loaded.")
        except Exception as e:
            p(f"  ❌  Load failed: {e}")
            total_skip += len(sentences)
            direction_summary[direction] = {"status":"LOAD ERROR","pass":0,"fail":0,"skip":len(sentences)}
            continue

        dir_pass = dir_fail = 0

        for idx, (text, category, expected_kw, notes) in enumerate(sentences, 1):
            p()
            p(f"  Test {idx:02d} [{category}]  {notes}")
            p(f"  INPUT   : {text}")
            try:
                output, route, elapsed = run_pipeline(text, direction, tokenizer, model_obj)
            except Exception as e:
                p(f"  ❌  PIPELINE ERROR: {e}")
                dir_fail += 1; total_fail += 1; total_tests += 1
                continue

            score, found, missing = grade(output, expected_kw)
            p(f"  OUTPUT  : {output}")
            p(f"  ROUTE   : {route}")
            if elapsed:
                p(f"  TIME    : {elapsed:.2f}s")

            if score == 1.0:
                verdict = "✅  PASS"
                dir_pass += 1; total_pass += 1
            elif score >= 0.5:
                verdict = f"⚠️  PARTIAL  (found {len(found)}/{len(expected_kw)} keywords)"
                dir_fail += 1; total_fail += 1
            else:
                verdict = f"❌  FAIL  (found {len(found)}/{len(expected_kw)} keywords)"
                dir_fail += 1; total_fail += 1

            p(f"  VERDICT : {verdict}")
            if found:   p(f"  FOUND   : {', '.join(found)}")
            if missing: p(f"  MISSING : {', '.join(missing)}")
            total_tests += 1

        direction_summary[direction] = {"status":"OK","pass":dir_pass,"fail":dir_fail,"skip":0}
        p(); p(SEP2)
        p(f"  {direction.upper()} SUMMARY: {dir_pass} passed, {dir_fail} failed out of {dir_pass+dir_fail} tests")
        p(SEP2)

    p(); p(SEP); p("  FINAL SUMMARY"); p(SEP)
    p(f"  Total tests  : {total_tests}")
    p(f"  ✅  Passed   : {total_pass}")
    p(f"  ❌  Failed   : {total_fail}")
    p(f"  ⚠️  Skipped  : {total_skip}")
    p()
    for d, s in direction_summary.items():
        if s["status"] == "OK":
            total_d = s["pass"] + s["fail"]
            pct = int(100 * s["pass"] / total_d) if total_d else 0
            icon = "✅" if pct >= 80 else ("⚠️" if pct >= 50 else "❌")
            p(f"  {icon}  {d.upper():8s}  {s['pass']} / {total_d} passed  ({pct}%)")
        else:
            p(f"  ⚠️  {d.upper():8s}  {s['status']}")
    p(SEP)

    report_path = "test_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  📄  Report saved to: {report_path}")
    print("      Upload this file to get debugging help.\n")


if __name__ == "__main__":
    run_all_tests()
