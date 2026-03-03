# Knowl Roadmap

## 🔧 Planned Tools

### 1. Structured Web Extractor
- **What:** Give Claude a URL + optional target schema → returns clean, structured data
- **Why:** `fetch_page` returns raw HTML; this would parse and extract only what's relevant
- **Use cases:**
  - Product pages (parts, prices, specs)
  - Boat listings
  - Marine forum threads
  - Any structured web data → context file
- **Status:** Planned

---

### 2. PDF / File Upload Extractor
- **What:** Upload a PDF or document → Claude parses and extracts key fields → saves to context
- **Why:** Many important documents (surveys, insurance, manuals, invoices) are PDFs
- **Use cases:**
  - Boat survey report → auto-populate deficiencies into `todo.md`
  - Insurance policy → pull policy number, term, coverage into `project.md`
  - Yamaha service manual → extract specs, torque values, service intervals
  - Invoices → log costs against todo items
- **Status:** Planned

---

### 3. Diff / Merge for Context Files
- **What:** When Claude updates a context file, show a line-by-line diff of what changed
- **Why:** Currently Claude just says "done!" — no visibility into what actually changed
- **Status:** Planned
