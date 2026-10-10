"""
NSE Corporate Bond Market (CBM) data scraper.
Downloads CBM bhavcopy + WDM (G-sec) bhavcopy + security master, day by day.
"""

import json
import logging
import random
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

from curl_cffi import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nse_scraper")

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
START_DATE = date(2019, 1, 1)
END_DATE = date(2026, 8, 31)
OUT_DIR = Path(r"C:\A - Personal\User\UnderGrad\MBFM Project\Datasets\raw_data")

REPORTS_BASE_URL = "https://www.nseindia.com/api/reports"

REPORT_CBM_BHAVCOPY = {
    "name": "CBM - Bhavcopy for the day",
    "type": "archives",
    "category": "debt",
    "section": "corporate-bonds",
}
REPORT_WDM_BHAVCOPY = {
    "name": "Bhavcopy file (csv)",
    "type": "archives",
    "category": "debt",
    "section": "debt-segment",
}
# Unconfirmed - update name/section once you've Inspected the real link
REPORT_CBM_SECURITY_MASTER = {
    "name": "CBM - Security Master",
    "type": "archives",
    "category": "debt",
    "section": "corporate-bonds",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/octet-stream,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-debt",
}

MIN_DELAY_SEC = 1.0
MAX_DELAY_SEC = 2.5
MAX_RETRIES = 3


def build_report_url(report: dict, trade_date: date, mode: str = "single") -> str:
    archives_json = json.dumps([report], separators=(",", ":"))
    archives_param = quote(archives_json, safe="")
    date_str = trade_date.strftime("%d-%b-%Y")
    return (
        f"{REPORTS_BASE_URL}?archives={archives_param}"
        f"&date={date_str}&type={report['section']}&mode={mode}"
    )


def make_session() -> requests.Session:
    s = requests.Session(impersonate="chrome")
    s.headers.update(HEADERS)
    r = s.get("https://www.nseindia.com/", timeout=15)
    r.raise_for_status()
    time.sleep(1.0)
    return s


def fetch_csv(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                content = r.content
                stripped = content.lstrip()[:1]
                if stripped in (b"{", b"["):
                    return False  # holiday / no-data response
                if len(content) > 100:
                    dest.write_bytes(content)
                    return True
                log.warning("Suspiciously small response (%d bytes) for %s", len(content), url)
            elif r.status_code == 404:
                return False
            else:
                log.warning("HTTP %s on attempt %d for %s", r.status_code, attempt, url)
        except requests.RequestException as e:
            log.warning("Error on attempt %d for %s: %s", attempt, url, e)
        if attempt < MAX_RETRIES:
            time.sleep(3 * attempt)
            session.cookies.clear()
            session.get("https://www.nseindia.com/all-reports-debt", timeout=15)
    return False


def business_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def download_report_range(report: dict, start: date, end: date,
                           out_dir: Path, subdir: str, prefix: str):
    dest_dir = out_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    days = list(business_days(start, end))
    log.info("Downloading '%s' for %d candidate trading days", report["name"], len(days))

    ok, missing = 0, 0
    for i, d in enumerate(days, 1):
        url = build_report_url(report, d)
        dest = dest_dir / f"{prefix}_{d.strftime('%Y%m%d')}.csv"

        if fetch_csv(session, url, dest):
            ok += 1
        else:
            missing += 1

        if i % 50 == 0:
            log.info("Progress: %d/%d days (%d ok, %d missing/holiday)", i, len(days), ok, missing)

        time.sleep(random.uniform(MIN_DELAY_SEC, MAX_DELAY_SEC))
        if i % 150 == 0:
            session = make_session()

    log.info("'%s' complete: %d files saved, %d missing/holidays in %s",
              report["name"], ok, missing, dest_dir)


def download_security_master(out_dir: Path):
    master_dir = out_dir / "security_master"
    master_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    dest = master_dir / "security_master_latest.csv"
    url = build_report_url(REPORT_CBM_SECURITY_MASTER, date.today())
    if fetch_csv(session, url, dest):
        log.info("Security master saved to %s", dest)
    else:
        log.warning(
            "Security master download failed (REPORT_CBM_SECURITY_MASTER is unconfirmed). "
            "Grab it manually from 'CBM - Security Master' on the Corporate Bonds tab of "
            "https://www.nseindia.com/all-reports-debt and save it to: %s",
            dest,
        )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_report_range(REPORT_CBM_BHAVCOPY, START_DATE, END_DATE, OUT_DIR, "cbm_daily", "CBM")
    download_report_range(REPORT_WDM_BHAVCOPY, START_DATE, END_DATE, OUT_DIR, "gsec_daily", "GSEC")
    download_security_master(OUT_DIR)
    log.info("Done. Raw files are under %s", OUT_DIR)