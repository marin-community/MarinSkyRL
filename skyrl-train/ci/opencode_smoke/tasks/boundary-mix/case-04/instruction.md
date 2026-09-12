Use separate shell calls and retain the task goal after any context compaction. First run `seq 1 16000` so the tool produces enough output to force context summarization. After it returns, run `printf 'summary-survived\n' > /app/proof.txt` in a new call. Then run `cat /app/proof.txt` in another new call and finish only after seeing the marker.

