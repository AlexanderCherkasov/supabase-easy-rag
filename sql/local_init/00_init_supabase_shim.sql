-- ============================================================================
-- Local PostgreSQL Scaffolding for Standalone Postgres / Docker / CI
-- (This file is for local testing only; NOT needed on real Supabase projects)
-- ============================================================================

-- 1. Create standard Supabase roles if they do not exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        CREATE ROLE anon NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        CREATE ROLE service_role NOLOGIN;
    END IF;
END;
$$;

-- 2. Create auth schema & users table if not already present
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'auth' AND table_name = 'users') THEN
        CREATE SCHEMA IF NOT EXISTS auth;
        CREATE TABLE IF NOT EXISTS auth.users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    END IF;
END;
$$;

-- 3. Create standard auth.uid() and auth.jwt() functions if not already present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'auth' AND p.proname = 'uid'
    ) THEN
        CREATE OR REPLACE FUNCTION auth.uid()
        RETURNS UUID
        LANGUAGE sql STABLE
        AS $func$
            SELECT COALESCE(
                nullif(current_setting('request.jwt.claim.sub', true), ''),
                (nullif(current_setting('request.jwt.claims', true), '')::jsonb ->> 'sub')
            )::uuid;
        $func$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'auth' AND p.proname = 'jwt'
    ) THEN
        CREATE OR REPLACE FUNCTION auth.jwt()
        RETURNS JSONB
        LANGUAGE sql STABLE
        AS $func$
            SELECT COALESCE(
                nullif(current_setting('request.jwt.claims', true), '')::jsonb,
                jsonb_build_object(
                    'sub', current_setting('request.jwt.claim.sub', true),
                    'role', current_setting('request.jwt.claim.role', true)
                )
            );
        $func$;
    END IF;
END;
$$;
