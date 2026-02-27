import streamlit as st
import pandas as pd
import dns.resolver
import io
import re

st.set_page_config(page_title="PropStream → Instantly CSV", page_icon="📄", layout="wide")

st.title("PropStream → Instantly CSV Builder")

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "outlook.com",
    "icloud.com",
    "comcast.net",
    "att.net",
    "me.com",
    "msn.com",
    "live.com",
    "mac.com",
    "verizon.net",
    "sbcglobal.net",
    "cox.net",
    "proton.me",
    "protonmail.com",
    "gmx.com",
}

DISPOSABLE_DOMAINS = {
    "mailinator.com",
    "guerrillamail.com",
    "tempmail.com",
    "throwaway.email",
    "yopmail.com",
    "sharklasers.com",
    "guerrillamailblock.com",
    "grr.la",
    "dispostable.com",
    "trashmail.com",
    "trashmail.me",
    "fakeinbox.com",
    "temp-mail.org",
    "10minutemail.com",
    "getnada.com",
    "maildrop.cc",
    "harakirimail.com",
    "mailnesia.com",
    "binkmail.com",
    "safetymail.info",
}

# Output columns — Instantly-ready format
OUTPUT_COLUMNS = [
    "email",
    "first_name",
    "last_name",
    "phone",
    "company_name",
    "property_address",
    "property_city",
    "property_state",
    "property_zip",
    "property_full_address",
    "mail_address",
    "mail_city",
    "mail_state",
    "mail_zip",
    "mail_full_address",
]


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def _s(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _zip(value: object) -> str:
    s = _s(value)
    if not s:
        return ""
    s_lower = s.lower()
    if s_lower in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"-?\d+\.0", s):
        return s.split(".", 1)[0]
    return s


def _phone(value: object) -> str:
    digits = re.sub(r"\D+", "", _s(value))
    return digits


def _normalize_street(address: str) -> str:
    """Normalize street address for grouping (one lead per property)."""
    a = _s(address).lower()
    if not a:
        return ""
    a = a.replace(",", " ")
    a = re.sub(r"\s+", " ", a).strip()
    a = re.sub(r"\s+(#|apt|apartment|unit|ste|suite)\s*\w+\s*$", "", a).strip()
    return a


# ---------------------------------------------------------------------------
# Email validation helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _is_valid_syntax(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


@st.cache_data(show_spinner=False, ttl=3600)
def _domain_has_mx(domain: str) -> bool:
    try:
        answers = dns.resolver.resolve(domain, "MX", lifetime=5)
        return len(answers) > 0
    except Exception:
        return False


def _is_disposable(domain: str) -> bool:
    return domain in DISPOSABLE_DOMAINS


def validate_email(email: str) -> tuple[bool, str]:
    """
    Free verification: syntax → disposable check → MX lookup.
    Returns (is_valid, reason).
    """
    email = email.strip().lower()
    if not email:
        return False, "empty"

    if not _is_valid_syntax(email):
        return False, "bad_syntax"

    domain = email.split("@")[-1]

    if _is_disposable(domain):
        return False, "disposable"

    if not _domain_has_mx(domain):
        return False, "no_mx"

    return True, "ok"


# ---------------------------------------------------------------------------
# Email extraction & ranking
# ---------------------------------------------------------------------------

def _extract_emails_ordered(contact_row: pd.Series) -> list[str]:
    """Extract emails from Email 1→4 columns in PropStream priority order."""
    emails: list[str] = []

    explicit_cols = [f"Email {i}" for i in range(1, 10)]
    has_explicit = any(c in contact_row for c in explicit_cols)

    if has_explicit:
        for col in explicit_cols:
            if col not in contact_row:
                continue
            v = _s(contact_row[col]).lower()
            if v and "@" in v and not v.startswith("@") and not v.endswith("@"):
                emails.append(v)
    else:
        for col, value in contact_row.items():
            if "email" not in str(col).lower():
                continue
            v = _s(value).lower()
            if not v:
                continue
            parts = re.split(r"[,\s;]+", v)
            for p in parts:
                p = p.strip().lower()
                if p and "@" in p and not p.startswith("@") and not p.endswith("@"):
                    emails.append(p)

    seen: set[str] = set()
    out: list[str] = []
    for e in emails:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _score_email(email: str) -> int:
    """Higher = better. +10 personal domain, +5 valid MX, -100 if invalid."""
    email = email.strip().lower()
    valid, _ = validate_email(email)
    if not valid:
        return -100

    score = 5
    domain = email.split("@")[-1]
    if domain in PERSONAL_EMAIL_DOMAINS:
        score += 10
    return score


def _pick_best_email_validated(emails: list[str]) -> tuple[str, str]:
    """Pick the best email from the ordered list. Returns (email, reason)."""
    if not emails:
        return "", "no_emails"

    best_email = ""
    best_score = -999
    best_reason = "no_valid_emails"

    for email in emails:
        score = _score_email(email)
        if score > best_score:
            best_score = score
            best_email = email
            _, reason = validate_email(email)
            best_reason = reason

    if best_score <= -100:
        return "", best_reason

    return best_email, "ok"


# ---------------------------------------------------------------------------
# Phone helpers
# ---------------------------------------------------------------------------

def _is_dnc(value: object) -> bool:
    v = _s(value).strip().lower()
    return v in {"dnc", "yes", "true", "1", "y"}


def _pick_best_phone(contact_row: pd.Series) -> str:
    """Pick first non-DNC phone from Phone 1→5."""
    phone_cols = [f"Phone {i}" for i in range(1, 6)]
    if any(c in contact_row for c in phone_cols):
        for i in range(1, 6):
            phone = _phone(contact_row.get(f"Phone {i}", ""))
            if not phone or len(phone) < 10:
                continue
            if _is_dnc(contact_row.get(f"Phone {i} DNC", "")):
                continue
            return phone
        return ""

    # Fallback: scan any phone-like column
    for col, value in contact_row.items():
        col_l = str(col).lower()
        if "phone" not in col_l and col_l not in {"mobile", "landline"}:
            continue
        p = _phone(value)
        if p and len(p) >= 10:
            return p
    return ""


def _get(row: pd.Series, candidates: list[str]) -> str:
    """Return the first non-empty value from candidate column names."""
    for c in candidates:
        if c in row:
            v = _s(row.get(c, ""))
            if v:
                return v
    return ""


# ---------------------------------------------------------------------------
# Core: one lead per property from the contact file alone
# ---------------------------------------------------------------------------

def build_leads(
    contacts_df: pd.DataFrame,
    *,
    run_verification: bool = True,
    progress_callback=None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Process a PropStream contact/skip-trace export into Instantly-ready leads.

    Key behavior:
      - ONE lead per property address (best email wins)
      - Email validated: syntax + disposable + MX lookup
      - Deduped by email at the end (same person, multiple properties → one row)
    """
    contacts_df = contacts_df.copy()

    # Normalize the property address for grouping
    contacts_df["_norm_addr"] = contacts_df.apply(
        lambda r: _normalize_street(
            _get(r, ["Street Address", "Property Address", "Address"])
        ),
        axis=1,
    )

    # Group contacts by property address
    grouped = contacts_df.groupby("_norm_addr", sort=False)

    stats = {
        "total_contacts": len(contacts_df),
        "unique_properties": 0,
        "contacts_evaluated": 0,
        "emails_bad_syntax": 0,
        "emails_disposable": 0,
        "emails_no_mx": 0,
        "emails_valid": 0,
        "skipped_no_email": 0,
        "final_rows": 0,
        "deduped_emails": 0,
    }

    rows: list[dict[str, str]] = []
    groups = [(addr, group) for addr, group in grouped if addr]
    total = len(groups)
    stats["unique_properties"] = total

    for idx, (addr_key, group) in enumerate(groups):
        if progress_callback and idx % 50 == 0:
            progress_callback(idx / max(total, 1))

        best_candidate = None  # (email, phone, first, last, company, score, crow)

        for _, crow in group.iterrows():
            stats["contacts_evaluated"] += 1
            emails = _extract_emails_ordered(crow)

            if run_verification:
                email, reason = _pick_best_email_validated(emails)
                if reason == "bad_syntax":
                    stats["emails_bad_syntax"] += 1
                elif reason == "disposable":
                    stats["emails_disposable"] += 1
                elif reason == "no_mx":
                    stats["emails_no_mx"] += 1
            else:
                email = ""
                for e in emails:
                    domain = e.split("@")[-1].lower()
                    if domain in PERSONAL_EMAIL_DOMAINS:
                        email = e
                        break
                if not email and emails:
                    email = emails[0]

            if not email:
                stats["skipped_no_email"] += 1
                continue

            stats["emails_valid"] += 1
            score = _score_email(email) if run_verification else 0
            phone = _pick_best_phone(crow)
            first_name = _get(crow, ["First Name", "Owner 1 First Name", "FirstName"])
            last_name = _get(crow, ["Last Name", "Owner 1 Last Name", "LastName"])
            company_name = _get(crow, ["Company Name", "Company"])

            if best_candidate is None or score > best_candidate[5]:
                best_candidate = (email, phone, first_name, last_name, company_name, score, crow)

        if best_candidate is None:
            continue

        email, phone, first_name, last_name, company_name, _, crow = best_candidate

        # Property address (from contact file's Street Address / City / State / Zip)
        prop_addr = _get(crow, ["Street Address", "Property Address", "Address"])
        prop_city = _get(crow, ["City"])
        prop_state = _get(crow, ["State"])
        prop_zip = _zip(_get(crow, ["Zip"]))
        prop_full = ", ".join(p for p in [prop_addr, prop_city, prop_state, prop_zip] if p)

        # Owner mailing address
        mail_addr = _get(crow, ["Mail Street Address", "Mailing Address", "Mail Address"])
        mail_city = _get(crow, ["Mail City"])
        mail_state = _get(crow, ["Mail State"])
        mail_zip = _zip(_get(crow, ["Mail Zip"]))
        mail_full = ", ".join(p for p in [mail_addr, mail_city, mail_state, mail_zip] if p)

        rows.append({
            "email": email.lower(),
            "first_name": _s(first_name),
            "last_name": _s(last_name),
            "phone": _s(phone),
            "company_name": _s(company_name),
            "property_address": _s(prop_addr),
            "property_city": _s(prop_city),
            "property_state": _s(prop_state),
            "property_zip": _s(prop_zip),
            "property_full_address": _s(prop_full),
            "mail_address": _s(mail_addr),
            "mail_city": _s(mail_city),
            "mail_state": _s(mail_state),
            "mail_zip": _s(mail_zip),
            "mail_full_address": _s(mail_full),
        })

    if progress_callback:
        progress_callback(1.0)

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Deduplicate by email (same person owns multiple properties → one row)
    pre_dedup = len(out)
    out = out.drop_duplicates(subset=["email"], keep="first")
    stats["deduped_emails"] = pre_dedup - len(out)
    stats["final_rows"] = len(out)

    out = out.fillna("")
    return out, stats


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.caption(
    "Upload your PropStream **Contact / Skip Trace export** (CSV or Excel). "
    "The app picks **one lead per property** with the best verified email, "
    "ready to upload straight into Instantly."
)

uploaded_file = st.file_uploader(
    "Upload contact export (CSV or Excel)",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
)


def _read_file(uploaded_file) -> pd.DataFrame:
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
        return df.fillna("")
    df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return df.fillna("")


if uploaded_file is not None:
    try:
        contacts_df = _read_file(uploaded_file)
    except Exception as e:
        st.error(
            "Couldn't read the file. "
            "This app supports **.csv** and **.xlsx** (Excel) exports."
        )
        st.code(str(e)[:2000])
        st.stop()

    # Quick sanity check: does this look like a contact export?
    cols_lower = [c.lower() for c in contacts_df.columns]
    has_email = any("email" in c for c in cols_lower)
    has_phone = any("phone" in c for c in cols_lower)

    if not has_email and not has_phone:
        st.error(
            "This doesn't look like a PropStream contact/skip-trace export — "
            "no Email or Phone columns found. Make sure you're uploading the "
            "**Contact Export**, not the Property Export."
        )
        st.stop()

    st.success(f"Loaded **{len(contacts_df):,}** contacts with **{len(contacts_df.columns)}** columns.")

    run_verification = st.checkbox(
        "Verify emails (MX lookup — recommended)",
        value=True,
        help="Checks each email domain has a working mail server. Adds ~30-60s for large files but removes dead emails.",
    )

    st.divider()

    progress_bar = st.progress(0, text="Building leads…")

    def _update_progress(pct: float):
        progress_bar.progress(min(pct, 1.0), text=f"Processing… {int(pct * 100)}%")

    output_df, stats = build_leads(
        contacts_df=contacts_df,
        run_verification=run_verification,
        progress_callback=_update_progress,
    )

    progress_bar.progress(1.0, text="Done!")

    # Stats
    st.subheader("Build Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total contacts", f"{stats['total_contacts']:,}")
    c2.metric("Unique properties", f"{stats['unique_properties']:,}")
    c3.metric("Contacts with valid email", f"{stats['emails_valid']:,}")
    c4.metric("Final leads", f"{stats['final_rows']:,}")

    if run_verification:
        st.caption("Email verification breakdown:")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("No email on contact", f"{stats['skipped_no_email']:,}")
        v2.metric("Bad syntax", f"{stats['emails_bad_syntax']:,}")
        v3.metric("Disposable", f"{stats['emails_disposable']:,}")
        v4.metric("Dead domain (no MX)", f"{stats['emails_no_mx']:,}")

    if stats["deduped_emails"] > 0:
        st.caption(
            f"Removed **{stats['deduped_emails']:,}** duplicate emails "
            f"(same person owns multiple properties — kept first)."
        )

    st.subheader("Output Preview")
    st.dataframe(output_df, use_container_width=True, height=520)

    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    base_name = uploaded_file.name.rsplit(".", 1)[0]
    st.download_button(
        label="Download CSV for Instantly",
        data=csv_data,
        file_name=f"{base_name}_instantly.csv",
        mime="text/csv",
        use_container_width=True,
    )
