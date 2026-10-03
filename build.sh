#!/usr/bin/env bash
set -euo pipefail

# Preview deployments must not modify the production database.
if [[ "${CF_PAGES_BRANCH:-}" == "main" ]]; then
  : "${D1_DATABASE_NAME:?Set D1_DATABASE_NAME in Cloudflare Pages production build variables}"
  npx --yes wrangler d1 migrations apply "$D1_DATABASE_NAME" --remote --config wrangler.d1.jsonc
fi

# This is a plain static Pages project; the files in the repository root are
# the deployment output. Pages Functions are picked up from ./functions.
exit 0
