# Project conventions

## Commit messages

Prose theme, not comma-chained sub-feature enumeration. Lead with what was built or done in plain language, then headline numbers, then cost / what didn't work, then what's next. Casual voice is welcome; a light pun on a class name when it's natural ("smash gets smashed", "push surprisingly grabs") is in voice. Avoid AI-flavoured ceremony ("comprehensive", "robust", "leverages").

`%` in commit prose is acceptable shorthand for percentage points; don't pedant pp vs %. AU spelling otherwise (mislabelled, normalised, behaviour).

Reference example, in voice:

> Min-F1 focal loss (CDB-F1) built and tested. Run on nosides taxonomy. Mean wrist_smash lifts 4%; push surprisingly grabs +6.7% too. Overall range tightens. Smash gets...smashed (-5.5%). Macro at 0.75 / min 0.49 (wrist smash, still). Next trying dropping gamma to see if loss is getting swung too hard by single hard samples that might be mislabelled.

What that pattern carries:
- one-sentence "what was built / run" opener
- two or three sentences of headline result (lift, surprise, cost)
- one short clause for the closing-state metrics
- one sentence on what comes next and why

Do not enumerate every file touched, every test added, every flag flipped. The diff shows the what; the message carries the why and the result.
