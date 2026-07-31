# Iris self-service budget bumps

Users can increase their own Iris budget with:

```bash
uv run iris --cluster=cw-rno2a user budget set benjaminfeuer --limit 400000 --max-band interactive
```

The change does not survive a controller restart.
