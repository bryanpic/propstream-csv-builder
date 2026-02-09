# PropStream CSV Builder (Streamlit)

Super-simple Streamlit tool that merges two PropStream exports into a clean **lead CSV**:

1. **Contact / Skip Trace Export** (names, phone numbers, up to 4 emails)
2. **Property Export** (property details, MLS data, estimated value/equity)

Both exports can be **CSV or Excel (`.xlsx`)**.

## Run

```bash
pip3 install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501` or similar).

## Use

1. Upload **both exports at once** (2 files total). Order doesn't matter.
2. The app shows the output table and you can download it.

## Output columns

The generated CSV includes:

- `email` (one prioritized email per contact, lowercased)
- `first_name`, `last_name`, `company_name`
- `phone` (first available phone from the contact/skip-trace export)
- property fields (`property_*`, `mls_*`, `est_equity`)

## Deploy on Streamlit Community Cloud

**Don’t use the “Deploy” button in Cursor/VS Code** — it often shows “not connected to a remote.” Use the website instead:

1. Open **https://share.streamlit.io** and sign in with GitHub.
2. Click **“New app”** (or “Create app”).
3. Choose:
   - **Repository:** `bryanpic/propstream-csv-builder`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

If the repo doesn’t appear, go to **Settings → Linked accounts** and (re)connect GitHub so Streamlit can see your repos.
