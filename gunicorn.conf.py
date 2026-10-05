# Gunicorn configuration file
# This is automatically loaded by Gunicorn.

workers = 1
timeout = 120
bind = "0.0.0.0:10000" # This will be overridden by Render's $PORT binding automatically
