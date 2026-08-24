#!/usr/bin/env python3
"""
Diagnose NASA Earthdata / GES DISC access for the IMERG smoke test WITHOUT exposing
the token.

Run this on the HOST where EARTHDATA_TOKEN is set and there is network egress to
nasa.gov. (The dev sandbox is air-gapped and cannot reach NASA, so this cannot be
run there.)

    cd backend
    python scripts/diagnose_earthdata_access.py
    # options:  --product early|late|final   --days N

It answers three questions that a plain "HTTP 401" cannot:
  1. Is the request even reaching GES DISC with the Bearer token attached?
  2. Does the redirect chain end at a URS "authorize this application" page
     (i.e. the "NASA GESDISC DATA ARCHIVE" app is NOT approved for this account)?
  3. Does GES DISC report an invalid/expired token vs an authorization/EULA problem
     (read from the WWW-Authenticate header)?
  4. Does the granule's .dds show the ACTUAL V07 variable name + dimension order,
     so a rejected constraint (HTTP 400) can be matched to the real dataset?

SAFETY: the token value is never printed -- only a short, non-reversible sha256
fingerprint and its length. oauth codes/state and cookie values are redacted from
every URL and header shown. It downloads no science data (tiny subset constraint,
manual redirect following) and changes nothing on disk or in the account.
"""
import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, urlunparse

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Prefer the PRODUCTION trust logic + index helper so this diagnostic exercises the
# exact same code the smoke test relies on. Fall back to local copies if imported
# from outside the repo (e.g. an air-gapped sandbox where the app package deps are
# absent) so the module still imports for offline self-tests.
try:
    from app.services.weather_ingestion import _EarthdataAuthSession, get_imerg_indices
    from app.core.config_states import get_pilot_aoi_bounds
    _REPO_IMPORTS = True
    _IMPORT_ERROR = None
except Exception as _imp_err:  # pragma: no cover
    _EarthdataAuthSession = None
    get_imerg_indices = None
    get_pilot_aoi_bounds = None
    _REPO_IMPORTS = False
    _IMPORT_ERROR = _imp_err

_TRUSTED_HOSTS = ("earthdata.nasa.gov", "eosdis.nasa.gov")


def _local_is_trusted(host):
    host = (host or "").lower()
    return any(host == d or host.endswith("." + d) for d in _TRUSTED_HOSTS)


# use production's trust function when available (so a bug there would show up here)
_is_trusted = (
    _EarthdataAuthSession._is_trusted
    if (_REPO_IMPORTS and _EarthdataAuthSession is not None)
    else _local_is_trusted
)


def _host(url):
    return (urlparse(url).hostname or "").lower()


def _redact(text, token):
    """Remove the token and any oauth secrets from a string before printing."""
    if not text:
        return text
    if token and token in text:
        text = text.replace(token, "<TOKEN_REDACTED>")
    text = re.sub(
        r"(?i)\b(code|state|access_token|id_token|refresh_token|token|client_secret)=[^&\s\"']+",
        r"\1=<REDACTED>",
        text,
    )
    return text


def _safe_url(url):
    """Show scheme://host/path only -- drop query + fragment (may carry oauth codes)."""
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))


def _print_interesting_headers(resp, token):
    for key in ("WWW-Authenticate", "Server", "Content-Type", "Content-Length"):
        if key in resp.headers:
            print("      %s: %s" % (key, _redact(resp.headers[key], token)))
    if "Location" in resp.headers:
        loc = urljoin(resp.url, resp.headers["Location"])
        print("      Location: %s   (host=%s)" % (_safe_url(loc), _host(loc)))
    set_cookie = resp.headers.get("Set-Cookie")
    if set_cookie:
        # names only -- never cookie values
        names = sorted(set(re.findall(r"(?:^|,\s*)([A-Za-z0-9_.\-]+)=", set_cookie)))
        print("      Set-Cookie: %d cookie(s) [%s]" % (len(names), ", ".join(names)))


def _granule_url(product_key, date):
    """Granule (dataset) URL for one day, WITHOUT response suffix or constraint.
    Mirrors weather_ingestion._fetch_imerg_day's product/filename selection."""
    y, m, d = date.strftime("%Y"), date.strftime("%m"), date.strftime("%d")
    if product_key == "final":
        product = "GPM_3IMERGDF.07"
        filename = "3B-DAY.MS.MRG.3IMERG.%s%s%s-S000000-E235959.V07B.nc4" % (y, m, d)
    elif product_key == "late":
        product = "GPM_3IMERGDL.07"
        filename = "3B-DAY-L.MS.MRG.3IMERG.%s%s%s-S000000-E235959.V07B.nc4" % (y, m, d)
    else:
        product = "GPM_3IMERGDE.07"
        filename = "3B-DAY-E.MS.MRG.3IMERG.%s%s%s-S000000-E235959.V07B.nc4" % (y, m, d)
    return "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/%s/%s/%s/%s" % (
        product, y, m, filename,
    )


def build_opendap_url(product_key, date, bounds):
    """Reproduce EXACTLY the data URL that weather_ingestion._fetch_imerg_day builds.
    V07 uses 'precipitation' (V06 'precipitationCal' -> HTTP 400 'variable not found')."""
    lat_min, lat_max, lon_min, lon_max = get_imerg_indices(bounds)
    query = "?precipitation[0:0][%d:%d][%d:%d]" % (lon_min, lon_max, lat_min, lat_max)
    return _granule_url(product_key, date) + ".nc4" + query


def build_dds_url(product_key, date):
    """.dds descriptor of the SAME granule (metadata only -- no science data)."""
    return _granule_url(product_key, date) + ".dds"


def probe_dataset_descriptor(product_key, date, token):
    """
    Fetch the granule .dds (metadata only -- NO science data) from the real V07
    endpoint and print every declaration line mentioning precipitation, so the
    ACTUAL variable name and dimension order can be read off the live dataset
    instead of guessed. This is what turns a rejected constraint into a verified
    one. Uses the production _EarthdataAuthSession so NASA redirects keep auth.
    """
    if _EarthdataAuthSession is None:
        print("  (.dds probe skipped: app package not importable here)")
        return
    dds_url = build_dds_url(product_key, date)
    print("Dataset descriptor (.dds -- metadata only, NO data download):")
    print("  GET %s" % _safe_url(dds_url))
    try:
        sess = _EarthdataAuthSession()
        if token:
            sess.headers.update({"Authorization": "Bearer %s" % token})
        resp = sess.get(dds_url, timeout=30)
    except requests.RequestException as exc:
        print("  request error: %s" % _redact(str(exc), token))
        return
    print("  -> HTTP %d" % resp.status_code)
    if resp.status_code != 200:
        snippet = _redact((resp.text or "")[:300].replace("\n", " ").strip(), token)
        if snippet:
            print("  body[:300]: %s" % snippet)
        return
    precip_lines = [ln.strip() for ln in (resp.text or "").splitlines()
                    if "precipitation" in ln.lower()]
    if precip_lines:
        print("  precipitation declaration(s) on the real endpoint:")
        for ln in precip_lines:
            print("     %s" % _redact(ln, token))
        print("  NOTE: the constraint's index order must match the order shown above.")
        print("        This code sends [time][lon][lat]; if the endpoint declares")
        print("        [time][lat][lon], swap the two index groups in the constraint.")
    else:
        print("  (no 'precipitation' line found; first 1500 chars of .dds:)")
        print(_redact((resp.text or "")[:1500], token))


def follow_chain(start_url, token, max_hops=8):
    """
    Walk the redirect chain by hand (allow_redirects=False) so we can see each hop and
    control the Authorization header exactly the way production does: keep it for
    NASA<->NASA hops, drop it for a hop to any other host.
    """
    session = requests.Session()  # cookie jar only; Authorization is managed per hop
    url = start_url
    send_auth = bool(token)
    flags = {
        "final_status": None,
        "authorize_app_page": False,
        "invalid_token": False,
        "auth_dropped_at_hop": None,
        "reached_gesdisc": False,
        "error": None,
    }

    for hop in range(1, max_hops + 1):
        headers = {"Authorization": "Bearer %s" % token} if send_auth else {}
        try:
            resp = session.get(url, headers=headers, allow_redirects=False, timeout=30)
        except requests.RequestException as exc:
            flags["error"] = "%s: %s" % (type(exc).__name__, _redact(str(exc), token))
            print("  hop %d: GET host=%s -> REQUEST ERROR: %s" % (hop, _host(url), flags["error"]))
            return flags

        if _host(url).endswith("gesdisc.eosdis.nasa.gov"):
            flags["reached_gesdisc"] = True

        print("  hop %d: GET host=%s path=%s" % (hop, _host(url), urlparse(url).path[:90]))
        print("           -> HTTP %d   auth_sent=%s" % (
            resp.status_code, "yes" if "Authorization" in headers else "no"))
        _print_interesting_headers(resp, token)

        www_auth = resp.headers.get("WWW-Authenticate", "").lower()
        if "invalid_token" in www_auth or "expired" in www_auth:
            flags["invalid_token"] = True

        # URS showing an application-approval page => app not authorized for this account
        if (_host(url).endswith("earthdata.nasa.gov")
                and "/oauth/authorize" in urlparse(url).path
                and resp.status_code == 200):
            body = (resp.text or "")[:6000].lower()
            if ("authorize" in body or "approve" in body
                    or "not yet authorized" in body or "eula" in body):
                flags["authorize_app_page"] = True
                print("           !! URS returned an APPLICATION-AUTHORIZATION / EULA page.")
                print("              => the GES DISC application is NOT yet approved for this account.")

        is_redirect = resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308)
        if is_redirect:
            location = resp.headers.get("Location")
            if not location:
                print("           (redirect status with no Location header; stopping)")
                flags["final_status"] = resp.status_code
                return flags
            next_url = urljoin(url, location)
            if send_auth and _host(url) != _host(next_url):
                if _is_trusted(_host(url)) and _is_trusted(_host(next_url)):
                    pass  # NASA<->NASA: production keeps the header (the fix)
                else:
                    send_auth = False
                    flags["auth_dropped_at_hop"] = hop
                    print("           (Authorization DROPPED for next hop -> host=%s is non-NASA)"
                          % _host(next_url))
            url = next_url
            continue

        # terminal response
        flags["final_status"] = resp.status_code
        if resp.status_code >= 400:
            snippet = _redact((resp.text or "")[:300].replace("\n", " ").strip(), token)
            if snippet:
                print("           body[:300]: %s" % snippet)
        return flags

    print("  (stopped after %d hops)" % max_hops)
    return flags


def print_verdict(flags):
    print()
    print("=" * 66)
    print("VERDICT")
    print("=" * 66)
    fs = flags["final_status"]
    if flags["error"]:
        print("INCONCLUSIVE - network error before a terminal response:")
        print("  %s" % flags["error"])
        print("  Check egress/proxy/DNS to gpm1.gesdisc.eosdis.nasa.gov from this host.")
    elif fs == 200 and not flags["authorize_app_page"]:
        print("AUTH OK - GES DISC accepted the Bearer token (HTTP 200).")
        print("  If the smoke test still fails, the cause is NOT authentication:")
        print("  check granule availability for the chosen date/product, the OPeNDAP")
        print("  constraint, or NetCDF parsing (xarray/h5netcdf) instead.")
    elif flags["authorize_app_page"]:
        print("APPLICATION NOT AUTHORIZED - this is an account setting, not a code bug.")
        _print_authorize_remedy()
    elif flags["invalid_token"]:
        print("TOKEN INVALID or EXPIRED - GES DISC/URS reported invalid_token.")
        print("  Regenerate the token at https://urs.earthdata.nasa.gov/profile ->")
        print("  Generate Token, copy it whole (no quotes/newline), re-export EARTHDATA_TOKEN.")
    elif fs in (401, 403):
        print("HTTP %d after the Bearer token reached GES DISC." % fs)
        print("  The header IS being sent and preserved across the NASA redirect, so this")
        print("  is almost certainly an ACCOUNT AUTHORIZATION / EULA condition, not code:")
        _print_authorize_remedy()
    elif fs == 400:
        print("HTTP 400 - GES DISC FOUND the granule but REJECTED the OPeNDAP")
        print("  constraint expression (auth is fine; the request reached the data")
        print("  host). This is a URL/constraint construction bug, not auth:")
        print("   * variable name: V06 'precipitationCal' was renamed to")
        print("     'precipitation' in V07 -- a V06 name yields HTTP 400 here;")
        print("   * dimension order / index range: precipitation[time][lon][lat] --")
        print("     a transposed or out-of-range index also yields 400.")
        print("  Compare the constraint with the .dds declaration printed above.")
    elif fs == 404:
        print("HTTP 404 - the granule/URL was not found (NOT an auth failure).")
        print("  The chosen date likely isn't published yet for this product (latency:")
        print("  Early ~4h, Late ~14h, Final ~3.5 months). Re-run with --days or --product.")
    elif flags["auth_dropped_at_hop"] is not None:
        print("Authorization was dropped at hop %d (redirect to a non-NASA host)."
              % flags["auth_dropped_at_hop"])
        print("  If that host was actually a NASA Earthdata/EOSDIS host, the trust list in")
        print("  _EarthdataAuthSession needs to include it (a real code fix). Check the host")
        print("  printed above against earthdata.nasa.gov / eosdis.nasa.gov.")
    else:
        print("INCONCLUSIVE - terminal HTTP %s. See the hop trace above." % fs)


def _print_authorize_remedy():
    print("  Fix (once per account):")
    print("   1. Sign in at https://urs.earthdata.nasa.gov")
    print("   2. Profile -> Applications -> Authorized Apps")
    print("   3. Find 'NASA GESDISC DATA ARCHIVE' and click Approve (accept the EULA")
    print("      if prompted). See https://disc.gsfc.nasa.gov/information/documents")
    print("      ('Data Access' / Earthdata Login).")
    print("   4. Re-run this script; a healthy result ends in HTTP 200.")


def main():
    parser = argparse.ArgumentParser(description="Token-safe NASA GES DISC access diagnostic")
    parser.add_argument("--product", choices=("early", "late", "final"), default="early")
    parser.add_argument("--days", type=int, default=10,
                        help="probe a granule this many days in the past (default 10)")
    args = parser.parse_args()

    print("=" * 66)
    print("NASA Earthdata / GES DISC ACCESS DIAGNOSTIC (token is never printed)")
    print("=" * 66)

    if not _REPO_IMPORTS:
        print("ERROR: could not import the app package -- run this from the backend/ dir")
        print("       (cd backend && python scripts/diagnose_earthdata_access.py).")
        print("       import error: %r" % (_IMPORT_ERROR,))
        return 2

    token = os.environ.get("EARTHDATA_TOKEN")
    username = os.environ.get("EARTHDATA_USERNAME")
    if token:
        fp = hashlib.sha256(token.encode("utf-8", "replace")).hexdigest()[:12]
        print("EARTHDATA_TOKEN   : present (len=%d, sha256[:12]=%s)  # value not shown" % (len(token), fp))
        stripped = token.strip()
        if stripped != token:
            print("  WARNING: token has leading/trailing whitespace/newline -> re-export it trimmed.")
    else:
        print("EARTHDATA_TOKEN   : NOT SET")
    print("EARTHDATA_USERNAME: %s" % ("present" if username else "not set"))
    if not token:
        print("\nSet EARTHDATA_TOKEN and re-run.")
        return 2

    bounds = get_pilot_aoi_bounds("Sikkim")
    date = datetime.utcnow() - timedelta(days=args.days)
    url = build_opendap_url(args.product, date, bounds)
    print("Product           : %s" % args.product)
    print("Probe date (UTC)  : %s" % date.strftime("%Y-%m-%d"))
    print("Probe URL         : %s" % _redact(url, token))
    print("AOI bounds        : %s" % bounds)
    print()
    probe_dataset_descriptor(args.product, date, token)
    print()
    print("Redirect / auth chain (allow_redirects=False, one hop at a time):")
    flags = follow_chain(url, token)
    print_verdict(flags)
    # exit non-zero unless we proved a clean 200, so this is CI/script friendly
    return 0 if flags.get("final_status") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
