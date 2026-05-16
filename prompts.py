"""
prompts.py
==========
System prompt + per-ticker prompt template for the Stocks project.
Claude acts as a senior equity analyst applying the Investment Valuation Framework v2.

Usage:
    from prompts import SYSTEM_PROMPT, build_analysis_prompt
    messages = [
        {"role": "user", "content": build_analysis_prompt(ticker_data)}
    ]
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Defines Claude's identity, the full framework, and output format.
# Sent once per API session. Never changes between tickers.
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a senior equity analyst applying the Investment Valuation Framework v2.
Your job is to analyze publicly listed stocks step by step, showing every
calculation explicitly, explaining why each metric matters for the specific
company type, and delivering a clear Buy / Watch / Avoid verdict.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FRAMEWORK RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — CLASSIFY THE COMPANY
Sector: one of the 11 GICS sectors.
Financial Profile: one of these 8:
  1. Mature FCF Generator   — positive FCF 5+ years, stable, 0-10% growth
  2. Growth + FCF Positive  — revenue ≥15% growth, FCF already positive
  3. Pre-Profit High Growth — fast growth, FCF negative, reinvests all
  4. Dividend Payer         — consistent dividends 5+ years
  5. Asset-Heavy/Regulated  — banks, insurers, regulated utilities
  6. Cyclical Business      — earnings swing 30-50%+ with cycle
  7. Turnaround/Distressed  — declining revenue, restructuring
  8. REIT                   — real estate investment trust structure

STEP 2 — SELECT VALUATION METHODS
Choose 2-3 methods based on financial profile (NOT sector alone):
  - Mature FCF / Dividend:     P/OCF, DCF, DDM, EV/EBITDA
  - Growth + FCF:              EV/EBITDA, PEG, Reverse DCF, P/FCF
  - Pre-Profit:                EV/Revenue, Rule of 40, TAM analysis
  - Asset-Heavy (banks):       P/Book, ROE vs cost of equity, NIM
  - Asset-Heavy (utilities):   RAB, EV/EBITDA, Dividend Yield
  - Cyclical:                  Mid-cycle EV/EBITDA, normalised P/OCF
  - Turnaround:                EV/Revenue, liquidation/asset value
  - REIT:                      P/FFO, P/AFFO, NAV, Dividend Yield

STEP 3 — CALCULATE METRICS
Show the formula, plug in the numbers, state the result, compare to threshold.
For each metric explicitly write:
  Formula:  [formula name] = [components]
  Numbers:  [component A] / [component B] = [result]
  Threshold: healthy = [benchmark]
  Status:   ✓ healthy / ⚠ borderline / ✗ concern

STEP 4 — SCENARIOS (bear / base / bull)
Always run three scenarios varying the key assumption (growth rate or multiple).
Show the fair value output for each.

STEP 5 — MARGIN OF SAFETY
Calculate: Margin of Safety = (Fair Value - Current Price) / Fair Value × 100
State the required margin for this company type, and whether it is met.

STEP 6 — VERDICT
One of: BUY / WATCH / AVOID
State the primary reason in one sentence.
State the primary risk in one sentence.
State the catalyst needed in one sentence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Use plain text with ASCII separators (━━━) for section headers.
- Use ✓ / ⚠ / ✗ for metric status indicators.
- Show every number explicitly — no rounding without stating it.
- Never skip a calculation. If data is missing, state "DATA MISSING — using
  industry average of [X]" and explain why.
- If a metric is not applicable for this company type, state why and skip it.
- End every analysis with the VERDICT block in this exact format:

  ══════════════════════════════════════════════
  VERDICT: [BUY / WATCH / AVOID]
  Why:      [one sentence]
  Risk:     [one sentence]
  Catalyst: [one sentence]
  ══════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTOR-SPECIFIC METRIC RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Technology — Software/SaaS:
  Must calculate: ARR growth, NRR (if available), Gross Margin, Rule of 40,
  FCF Margin, SBC as % of revenue, CAC payback (if data available).
  Never use GAAP P/E as primary metric.

Technology — Semiconductors/Hardware:
  Must assess cycle position (upcycle/peak/downturn/trough/recovery).
  Must use mid-cycle normalised EBITDA, not current-year EBITDA.
  Never use peak-cycle P/E as primary metric.

Communication Services — Internet Platforms:
  Must calculate: ARPU, DAU/MAU ratio (if available), revenue mix
  (advertising vs subscriptions), FCF margin.
  For multi-business platforms: perform Sum-of-Parts before blended multiple.

Communication Services — Telecom:
  Must calculate: FCF after capex (not just EBITDA), Net Debt/EBITDA,
  dividend coverage ratio = FCF / Annual Dividend.
  Never rely on EBITDA alone — always deduct capex before assessing dividend safety.

Consumer Discretionary — Luxury/Brands:
  Must decompose revenue growth into volume vs price components.
  Must assess gross margin trend (expanding = pricing power; contracting = erosion).
  Must note DTC % of revenue and direction.

Consumer Discretionary — E-commerce:
  Must assess contribution margin (not just gross margin).
  For multi-business companies (Amazon): perform Sum-of-Parts.
  Must calculate take rate for marketplaces.

Consumer Discretionary — Restaurants/Experiences:
  Must calculate restaurant-level margin (not just corporate margin).
  Must note franchise % of system.
  Must assess SSS growth and decompose into price vs traffic if data available.

Financials — Banks:
  NEVER use EV/EBITDA or standard DCF. Use P/Book and ROE analysis.
  Must calculate: ROE, ROA, NIM, Tier 1 capital ratio, NPL ratio.

REITs:
  NEVER use EPS or standard P/E. Use P/FFO and P/AFFO.
  Must calculate: FFO/share, AFFO payout ratio, LTV, same-store NOI growth.

Cyclical companies (Energy, Materials, Industrials):
  NEVER value on current-year earnings if at cycle peak.
  Always calculate mid-cycle normalised metrics using 5-7 year averages.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PER-TICKER PROMPT BUILDER
# Injects real company data into a structured prompt.
# Call this function once per ticker to build the user message.
# ─────────────────────────────────────────────────────────────────────────────

def build_analysis_prompt(data: dict) -> str:
    """
    Build the per-ticker analysis prompt.

    Parameters
    ----------
    data : dict
        Dictionary of financial data for one company. All fields are optional
        — if a field is None or missing, Claude will note it and use
        industry benchmarks. See DATA SCHEMA below.

    Returns
    -------
    str
        The fully formatted user prompt to send to Claude API.

    DATA SCHEMA (all values optional — pass None if unavailable)
    ──────────────────────────────────────────────────────────────
    Required:
        ticker          str     e.g. "AAPL"
        company_name    str     e.g. "Apple Inc."
        current_price   float   e.g. 189.50
        currency        str     e.g. "USD"
        market          str     e.g. "NASDAQ", "TSX", "LSE"

    Classification (pre-classified by your code, or leave None for Claude):
        sector          str     GICS sector name
        financial_profile str   One of the 8 profiles above

    Income Statement (annual, last fiscal year unless noted):
        revenue             float   Total revenue
        revenue_prior_yr    float   Revenue one year prior (for growth calc)
        revenue_2yr_ago     float   Revenue two years prior
        gross_profit        float
        operating_income    float
        net_income          float
        ebitda              float
        ebit                float
        sbc                 float   Stock-based compensation
        interest_expense    float
        tax_rate            float   As decimal e.g. 0.21

    Balance Sheet:
        total_assets        float
        total_debt          float
        cash                float
        net_debt            float   (total_debt - cash)
        book_value_equity   float
        shares_outstanding  float   In millions
        tier1_capital_ratio float   Banks only
        npl_ratio           float   Banks only (non-performing loans %)

    Cash Flow Statement:
        operating_cash_flow float
        capex               float   As positive number
        free_cash_flow      float   (operating_cash_flow - capex)
        dividends_paid      float   As positive number
        buybacks            float   Share repurchases

    Valuation (market data):
        market_cap          float
        enterprise_value    float
        pe_ratio            float
        ev_ebitda           float
        ev_revenue          float
        price_to_book       float
        price_to_fcf        float
        dividend_yield      float   As decimal e.g. 0.035
        dividend_per_share  float

    Operational metrics (sector-specific — pass what you have):
        revenue_growth_3yr_cagr float   3-year revenue CAGR as decimal
        gross_margin            float   As decimal
        operating_margin        float   As decimal
        fcf_margin              float   As decimal
        roic                    float   Return on invested capital
        roe                     float   Return on equity
        roa                     float   Return on assets

    # SaaS-specific
        arr_growth              float   ARR growth YoY as decimal
        net_revenue_retention   float   NRR as decimal e.g. 1.15 = 115%
        rule_of_40_score        float   Calculated score

    # Telecom-specific
        arpu                    float   Average revenue per user (monthly)
        ebitda_after_capex      float   EBITDA minus capex
        dividend_coverage_ratio float   FCF / annual dividends

    # Restaurant-specific
        same_store_sales_growth float   SSS growth as decimal
        restaurant_level_margin float   As decimal
        franchise_pct           float   % of system franchised

    # REIT-specific
        ffo_per_share           float
        affo_per_share          float
        affo_payout_ratio       float
        occupancy_rate          float
        nav_per_share           float

    # Energy/Materials-specific
        mid_cycle_ebitda        float   7-10yr average EBITDA
        breakeven_price         float   Commodity break-even (oil: $/bbl)

    Context:
        analyst_notes       str     Optional — any extra context you want
                                    Claude to consider e.g. recent news,
                                    one-time items, management guidance
        discount_rate       float   Your required return e.g. 0.12 = 12%
        peer_ev_ebitda      float   Sector peer average EV/EBITDA multiple
        peer_gross_margin   float   Sector peer average gross margin
    """

    def fmt(val, pct=False, mult=1, decimals=2, prefix=""):
        """Format a number or return DATA MISSING."""
        if val is None:
            return "DATA MISSING"
        v = val * mult
        if pct:
            return f"{prefix}{v * 100:.{decimals}f}%"
        return f"{prefix}{v:,.{decimals}f}"

    def fmtb(val, prefix="$"):
        """Format large numbers as billions."""
        if val is None:
            return "DATA MISSING"
        if abs(val) >= 1e9:
            return f"{prefix}{val/1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"{prefix}{val/1e6:.2f}M"
        return f"{prefix}{val:,.2f}"

    # ── Extract fields with safe defaults ─────────────────────────────────
    ticker          = data.get("ticker", "UNKNOWN")
    company_name    = data.get("company_name", "Unknown Company")
    price           = data.get("current_price")
    currency        = data.get("currency", "USD")
    market          = data.get("market", "Unknown")
    sector          = data.get("sector", "Unknown — classify based on data")
    profile         = data.get("financial_profile", "Unknown — classify based on data")
    discount_rate   = data.get("discount_rate", 0.12)
    analyst_notes   = data.get("analyst_notes", "None provided.")

    # Income statement
    revenue         = data.get("revenue")
    rev_prior       = data.get("revenue_prior_yr")
    rev_2yr         = data.get("revenue_2yr_ago")
    gross_profit    = data.get("gross_profit")
    op_income       = data.get("operating_income")
    net_income      = data.get("net_income")
    ebitda          = data.get("ebitda")
    ebit            = data.get("ebit")
    sbc             = data.get("sbc")
    interest_exp    = data.get("interest_expense")
    tax_rate        = data.get("tax_rate", 0.21)

    # Balance sheet
    total_debt      = data.get("total_debt")
    cash            = data.get("cash")
    net_debt        = data.get("net_debt")
    book_equity     = data.get("book_value_equity")
    shares          = data.get("shares_outstanding")
    tier1           = data.get("tier1_capital_ratio")
    npl             = data.get("npl_ratio")

    # Cash flow
    ocf             = data.get("operating_cash_flow")
    capex           = data.get("capex")
    fcf             = data.get("free_cash_flow")
    dividends       = data.get("dividends_paid")
    buybacks        = data.get("buybacks")

    # Valuation
    market_cap      = data.get("market_cap")
    ev              = data.get("enterprise_value")
    pe              = data.get("pe_ratio")
    ev_ebitda       = data.get("ev_ebitda")
    ev_rev          = data.get("ev_revenue")
    ptb             = data.get("price_to_book")
    ptfcf           = data.get("price_to_fcf")
    div_yield       = data.get("dividend_yield")
    dps             = data.get("dividend_per_share")

    # Operational
    rev_cagr        = data.get("revenue_growth_3yr_cagr")
    gross_margin    = data.get("gross_margin")
    op_margin       = data.get("operating_margin")
    fcf_margin      = data.get("fcf_margin")
    roic            = data.get("roic")
    roe             = data.get("roe")
    roa             = data.get("roa")

    # Sector-specific
    arr_growth      = data.get("arr_growth")
    nrr             = data.get("net_revenue_retention")
    rule40          = data.get("rule_of_40_score")
    arpu            = data.get("arpu")
    ebitda_cx       = data.get("ebitda_after_capex")
    div_coverage    = data.get("dividend_coverage_ratio")
    sss             = data.get("same_store_sales_growth")
    rl_margin       = data.get("restaurant_level_margin")
    franchise_pct   = data.get("franchise_pct")
    ffo_ps          = data.get("ffo_per_share")
    affo_ps         = data.get("affo_per_share")
    affo_payout     = data.get("affo_payout_ratio")
    occupancy       = data.get("occupancy_rate")
    nav_ps          = data.get("nav_per_share")
    mid_ebitda      = data.get("mid_cycle_ebitda")
    breakeven       = data.get("breakeven_price")
    peer_ev_ebitda  = data.get("peer_ev_ebitda")
    peer_gm         = data.get("peer_gross_margin")

    prompt = f"""
Analyze {company_name} ({ticker}) using the Investment Valuation Framework v2.
Work through every step explicitly, showing all calculations with real numbers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPANY DATA — {ticker} | {company_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Market:           {market}
Currency:         {currency}
Current Price:    {fmt(price, prefix=currency + " ")}
Market Cap:       {fmtb(market_cap)}
Enterprise Value: {fmtb(ev)}

── INCOME STATEMENT ─────────────────────────────────────────────────────────
Revenue (LTM):            {fmtb(revenue)}
Revenue (prior year):     {fmtb(rev_prior)}
Revenue (2 years ago):    {fmtb(rev_2yr)}
Gross Profit:             {fmtb(gross_profit)}
EBITDA:                   {fmtb(ebitda)}
EBIT:                     {fmtb(ebit)}
Operating Income:         {fmtb(op_income)}
Net Income:               {fmtb(net_income)}
Stock-Based Compensation: {fmtb(sbc)}
Interest Expense:         {fmtb(interest_exp)}
Tax Rate:                 {fmt(tax_rate, pct=True)}

── BALANCE SHEET ─────────────────────────────────────────────────────────────
Total Debt:               {fmtb(total_debt)}
Cash & Equivalents:       {fmtb(cash)}
Net Debt:                 {fmtb(net_debt)}
Book Value of Equity:     {fmtb(book_equity)}
Shares Outstanding:       {fmt(shares)}M
Tier 1 Capital Ratio:     {fmt(tier1, pct=True)}
NPL Ratio:                {fmt(npl, pct=True)}

── CASH FLOW STATEMENT ────────────────────────────────────────────────────────
Operating Cash Flow:      {fmtb(ocf)}
Capital Expenditure:      {fmtb(capex)}
Free Cash Flow:           {fmtb(fcf)}
Dividends Paid:           {fmtb(dividends)}
Share Buybacks:           {fmtb(buybacks)}

── VALUATION MULTIPLES (current market pricing) ───────────────────────────────
P/E Ratio:                {fmt(pe)}x
EV/EBITDA:                {fmt(ev_ebitda)}x
EV/Revenue:               {fmt(ev_rev)}x
Price/Book:               {fmt(ptb)}x
Price/FCF:                {fmt(ptfcf)}x
Dividend Yield:           {fmt(div_yield, pct=True)}
Dividend Per Share:       {fmt(dps, prefix=currency + " ")}

── OPERATIONAL METRICS ────────────────────────────────────────────────────────
Revenue CAGR (3yr):       {fmt(rev_cagr, pct=True)}
Gross Margin:             {fmt(gross_margin, pct=True)}
Operating Margin:         {fmt(op_margin, pct=True)}
FCF Margin:               {fmt(fcf_margin, pct=True)}
ROIC:                     {fmt(roic, pct=True)}
ROE:                      {fmt(roe, pct=True)}
ROA:                      {fmt(roa, pct=True)}
Peer Avg EV/EBITDA:       {fmt(peer_ev_ebitda)}x
Peer Avg Gross Margin:    {fmt(peer_gm, pct=True)}

── SECTOR-SPECIFIC METRICS ────────────────────────────────────────────────────
ARR Growth (SaaS):              {fmt(arr_growth, pct=True)}
Net Revenue Retention (SaaS):   {fmt(nrr, pct=True)}
Rule of 40 Score (SaaS):        {fmt(rule40)}
ARPU Monthly (Telecom):         {fmt(arpu, prefix=currency + " ")}
EBITDA after Capex (Telecom):   {fmtb(ebitda_cx)}
Dividend Coverage Ratio:        {fmt(div_coverage)}x
Same-Store Sales Growth:        {fmt(sss, pct=True)}
Restaurant-Level Margin:        {fmt(rl_margin, pct=True)}
Franchise % of System:          {fmt(franchise_pct, pct=True)}
FFO Per Share (REIT):           {fmt(ffo_ps, prefix=currency + " ")}
AFFO Per Share (REIT):          {fmt(affo_ps, prefix=currency + " ")}
AFFO Payout Ratio (REIT):       {fmt(affo_payout, pct=True)}
Occupancy Rate (REIT):          {fmt(occupancy, pct=True)}
NAV Per Share (REIT):           {fmt(nav_ps, prefix=currency + " ")}
Mid-Cycle EBITDA (Cyclical):    {fmtb(mid_ebitda)}
Commodity Break-Even:           {fmt(breakeven, prefix="$")} /bbl

── ANALYSIS PARAMETERS ────────────────────────────────────────────────────────
Discount Rate (required return): {fmt(discount_rate, pct=True)}
Pre-classified Sector:           {sector}
Pre-classified Financial Profile: {profile}

── ANALYST NOTES ──────────────────────────────────────────────────────────────
{analyst_notes}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Now work through the full framework step by step:

STEP 1 — CLASSIFY
Confirm or correct the sector and financial profile.
Explain in 2-3 sentences why this classification is correct given the data.

STEP 2 — SELECT VALUATION METHODS
State which 2-3 methods you will use and why they are appropriate
for this specific sector + financial profile combination.
Explicitly state any methods that are NOT appropriate and why.

STEP 3 — CALCULATE EVERY METRIC
For each metric relevant to this company type:
  - Write the formula
  - Plug in the actual numbers from the data above
  - State the result
  - Compare to the healthy threshold for this company type
  - Mark ✓ healthy / ⚠ borderline / ✗ concern
If any data is missing, state "DATA MISSING — using [benchmark] because [reason]"

STEP 4 — THREE SCENARIOS
Bear case:   Use conservative growth assumption. Calculate fair value.
Base case:   Use most likely growth assumption. Calculate fair value.
Bull case:   Use optimistic but realistic growth assumption. Calculate fair value.
Show the full calculation for each scenario, not just the result.

STEP 5 — MARGIN OF SAFETY
Calculate margin of safety vs the base case fair value.
State whether it meets the required threshold for this company type.
Required thresholds:
  High quality (wide moat):    ≥ 15%
  Good quality (moderate moat): ≥ 20-25%
  Average / cyclical:          ≥ 30%
  Turnaround / distressed:     ≥ 40-50%

STEP 6 — VERDICT
BUY / WATCH / AVOID with one sentence each for:
  Why (primary reason)
  Risk (primary risk)
  Catalyst (what needs to happen)
"""
    return prompt.strip()


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_DATA_MICROSOFT = {
    "ticker": "MSFT",
    "company_name": "Microsoft Corporation",
    "current_price": 415.50,
    "currency": "USD",
    "market": "NASDAQ",
    "sector": "Technology",
    "financial_profile": "Mature FCF Generator",

    "revenue": 245_122_000_000,
    "revenue_prior_yr": 211_915_000_000,
    "revenue_2yr_ago": 198_270_000_000,
    "gross_profit": 171_006_000_000,
    "ebitda": 130_000_000_000,
    "ebit": 109_000_000_000,
    "operating_income": 109_000_000_000,
    "net_income": 88_136_000_000,
    "sbc": 9_611_000_000,
    "interest_expense": 1_527_000_000,
    "tax_rate": 0.18,

    "total_debt": 79_000_000_000,
    "cash": 80_000_000_000,
    "net_debt": -1_000_000_000,
    "book_value_equity": 206_000_000_000,
    "shares_outstanding": 7_432,

    "operating_cash_flow": 118_548_000_000,
    "capex": 44_482_000_000,
    "free_cash_flow": 74_066_000_000,
    "dividends_paid": 22_317_000_000,
    "buybacks": 17_254_000_000,

    "market_cap": 3_088_000_000_000,
    "enterprise_value": 3_087_000_000_000,
    "pe_ratio": 35.0,
    "ev_ebitda": 23.7,
    "ev_revenue": 12.6,
    "price_to_book": 15.0,
    "price_to_fcf": 41.7,
    "dividend_yield": 0.0072,
    "dividend_per_share": 3.00,

    "revenue_growth_3yr_cagr": 0.155,
    "gross_margin": 0.698,
    "operating_margin": 0.445,
    "fcf_margin": 0.302,
    "roic": 0.38,
    "roe": 0.42,
    "roa": 0.18,

    "arr_growth": 0.17,
    "net_revenue_retention": 1.18,
    "rule_of_40_score": 61.7,

    "peer_ev_ebitda": 20.5,
    "peer_gross_margin": 0.62,

    "discount_rate": 0.10,
    "analyst_notes": (
        "Azure cloud growing at 29% YoY. Copilot AI features being monetised "
        "across Microsoft 365 — early signals of ARPU uplift. Gaming segment "
        "(Activision Blizzard acquisition) adding revenue but diluting margins "
        "short-term. Net cash position despite large acquisition financing. "
        "Consensus analyst fair value range: $380-$480."
    ),
}


EXAMPLE_DATA_BCE = {
    "ticker": "BCE",
    "company_name": "BCE Inc.",
    "current_price": 33.20,
    "currency": "CAD",
    "market": "TSX",
    "sector": "Communication Services",
    "financial_profile": "Dividend Payer",

    "revenue": 24_482_000_000,
    "revenue_prior_yr": 24_186_000_000,
    "gross_profit": 10_200_000_000,
    "ebitda": 10_700_000_000,
    "ebit": 4_200_000_000,
    "operating_income": 4_200_000_000,
    "net_income": 1_200_000_000,
    "interest_expense": 2_100_000_000,
    "tax_rate": 0.27,

    "total_debt": 38_000_000_000,
    "cash": 850_000_000,
    "net_debt": 37_150_000_000,
    "book_value_equity": 14_000_000_000,
    "shares_outstanding": 912,

    "operating_cash_flow": 7_800_000_000,
    "capex": 5_100_000_000,
    "free_cash_flow": 2_700_000_000,
    "dividends_paid": 3_870_000_000,
    "buybacks": 0,

    "market_cap": 30_278_000_000,
    "enterprise_value": 67_428_000_000,
    "pe_ratio": 25.2,
    "ev_ebitda": 6.3,
    "ev_revenue": 2.75,
    "price_to_book": 2.16,
    "dividend_yield": 0.128,
    "dividend_per_share": 3.99,

    "revenue_growth_3yr_cagr": 0.02,
    "gross_margin": 0.416,
    "operating_margin": 0.172,
    "fcf_margin": 0.110,
    "roe": 0.086,
    "roa": 0.018,

    "arpu": 68.50,
    "ebitda_after_capex": 5_600_000_000,
    "dividend_coverage_ratio": 0.70,

    "peer_ev_ebitda": 7.2,

    "discount_rate": 0.09,
    "analyst_notes": (
        "BCE dividend coverage ratio is below 1.0x — FCF does not fully "
        "cover the annual dividend of ~$3.87B. BCE cut its dividend in Feb 2025 "
        "by 56%%. Net Debt/EBITDA at 3.47x — elevated but within guidance range. "
        "Fibre build capex expected to peak in 2025 and normalise by 2026-2027. "
        "Management has guided FCF recovery post-capex-peak. Wireless subscriber "
        "growth is positive; media segment (CTV) being restructured."
    ),
}


if __name__ == "__main__":
    # Quick test to verify prompt generation
    prompt = build_analysis_prompt(EXAMPLE_DATA_MICROSOFT)
    print("SYSTEM PROMPT LENGTH:", len(SYSTEM_PROMPT), "chars")
    print("USER PROMPT LENGTH:  ", len(prompt), "chars")
    print()
    print("─" * 60)
    print("FIRST 500 chars of user prompt:")
    print(prompt[:500])
    print("─" * 60)
    print("LAST 300 chars of user prompt:")
    print(prompt[-300:])
