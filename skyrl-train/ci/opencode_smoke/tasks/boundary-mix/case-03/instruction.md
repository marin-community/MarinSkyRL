Exercise the terminal transport using three separate shell calls. First run `printf '\377\376\375GARBAGE\n'` exactly; its output is intentionally invalid UTF-8. After it returns, run `printf 'garbage-survived\n' > /app/proof.txt` in a new call. Then run `cat /app/proof.txt` in a third call and finish.

