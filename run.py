import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
