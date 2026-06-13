import { createClient, type SupabaseClient } from '@supabase/supabase-js';

// Public, read-only access. The publishable key is safe to ship to the browser:
// every benchmark table is governed by row-level security with select-only
// policies, and all writes happen offline through the service-role loader.
const SUPABASE_URL =
  import.meta.env.VITE_SUPABASE_URL ?? 'https://urnnulxipkmkwpkhqkip.supabase.co';
const SUPABASE_ANON_KEY =
  import.meta.env.VITE_SUPABASE_ANON_KEY ?? 'sb_publishable_TSBB_g6z8Xgr4H3xp0Nqtg_FsyAm7NS';

let client: SupabaseClient | null = null;

export function getSupabase(): SupabaseClient | null {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  if (!client) {
    client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false },
    });
  }
  return client;
}
