# Research dashboard verification

The existing framed-sheet application direction was retained. Full prose uses
16px text with 1.8 line height, paragraph breaks, and no clipping containers.

Verified in the live in-app browser:
- Normal desktop view: paired passages, progress, method table and chart readable.
- 390px and 768px widths: document width stays within viewport; passage grid stacks
  into one column; prose remains 16px. Temporary viewport overrides were reset.
- Method, rewrite-version and held-out evaluation filters change the visible data.
- Review draft survives background polling; the temporary draft was cleared without
  saving invented ratings.
- Stop/resume retains saved rows and continues the next unfinished cell.
- Resume progressed from 9 to 11 complete cells while the page was open.
- Full JSON export contains passages and excludes private keys (API test).

Source lint: 0 failures, 0 warnings. The standalone browser inspector could not
launch Chromium (spawn EPERM); live browser DOM checks and screenshot inspection
were used instead. No passing inspector measurements are claimed.

Screenshot: `output/research-dashboard.jpg`. The dashboard remains a live experiment;
semantic watermarking, translation and human false-positive evaluation are pending.

Windows reader-lock failures found during resume were corrected with unique
temporary files and bounded atomic-replace retries. Exited workers now appear
as resumable instead of remaining falsely labelled running.
