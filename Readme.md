# EdgeRAG Implementation

## Setup
1. Create virtual env
python3 -m venv edgerag_release

2. Setup swap 
https://www.jetson-ai-lab.com/tips_ram-optimization.html

3. Install dependencies
pip3 install -r requirements.txt

4. Edit virtual environment to allow pre-installed Jetson packages from jetpack \
In edgerag_release/pyvenv.cfg \
Change include-system-site-packages = false \
To include-system-site-packages = true

5. Create Index
See gen_index/Readme.txt

6. Run test
See run_test/notes.txt
