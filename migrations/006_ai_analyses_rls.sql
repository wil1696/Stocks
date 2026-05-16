-- Migration: 006_ai_analyses_rls
-- Allow public (anon + authenticated) to read ai_analyses so the dashboard works.
-- Writes still require the service role key (used by save_analysis.py).

ALTER TABLE ai_analyses ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Public read access to ai_analyses" ON ai_analyses;
DROP POLICY IF EXISTS "Anon and authenticated can read ai_analyses" ON ai_analyses;

CREATE POLICY "Anon and authenticated can read ai_analyses"
    ON ai_analyses
    FOR SELECT
    TO anon, authenticated
    USING (true);

-- Ensure anon role has the underlying SELECT grant on the table
GRANT SELECT ON ai_analyses TO anon;
GRANT SELECT ON ai_analyses TO authenticated;
